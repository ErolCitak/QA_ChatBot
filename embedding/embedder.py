import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch # type: ignore

import config
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_community.vectorstores import Chroma, FAISS # type: ignore
from langchain.docstore.document import Document # type: ignore
from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore


import warnings
warnings.filterwarnings('ignore')

class Embedder:
    def __init__(self, model_name, data_dir, documents, chunk_length=1500, overlap=100, save_vec_db=False):
        print("Initializing Embedding Model")
        print("-"*20)

        self.my_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": self.my_device})
        self.data_dir = data_dir
        self.documents = [os.path.join(self.data_dir, document) for document in documents]
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.save_vec_db = save_vec_db

    def embed_text(self, text=""):
        embedding_vector = self.embedding_model.embed_query(text) # return List[float]

        return embedding_vector

    def pdf_data_loader(self, idx=-1):
        if idx == -1:
            return ValueError(f"A document must be chosen for QA, curr idx: {idx}")
        else:
            loader = PyPDFLoader(self.documents[idx])
            loaded_document = loader.load()

            return loaded_document
        
    def text_splitter(self, data):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_length,
            chunk_overlap=self.overlap,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(data)

        return chunks
    
    def create_vector_database(self, chunks, top_k_doc=5):

        vector_db = FAISS.from_documents(chunks, self.embedding_model)
        vector_db_retriever = vector_db.as_retriever(search_kwargs={"k": top_k_doc})


        if self.save_vec_db:
            db_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            db_path = os.path.join(config.DB_PATH, db_name)
            vector_db.save_local(db_path)

        return vector_db, vector_db_retriever
    
    def load_vector_database(self, db_name):

        vector_db = FAISS.load_local(db_name, embeddings=self.embedding_model)
        vector_db_retriever = vector_db.as_retriever(search_kwargs={"search_type": "mmr", "k": 3})

        return vector_db, vector_db_retriever

    

if __name__ == "__main__":

    ## Embedder model name is taken from config.py
    embedder_name = config.EMBED_MODEL
    data_dir = config.DATA_DIR
    documents = config.DOCUMENTS

    # Embedder class is initializing
    embedder = Embedder(embedder_name, data_dir, documents, chunk_length=1000, overlap=100)

    #########################
    ### Sample Text Embedding
    ######################### 

    sample_text = "This is a sample text to verify embed query works well."
    sample_embedding_vector = embedder.embed_text(sample_text)

    print(f"Shape of the embedding: {len(sample_embedding_vector)}")
    print("-"*20)

    ##############################################
    ### Document Loading, Splitting and DB Storage
    ############################################## 
    # Select which document to use (e.g., first one)
    doc_index = 0

    # Step 1: Load the selected PDF document
    loaded_doc = embedder.pdf_data_loader(doc_index)
    print(f"Loaded Document: {documents[doc_index]}")

    # Step 2: Split the document into text chunks
    chunks = embedder.text_splitter(loaded_doc)
    print(f"Number of chunks created: {len(chunks)}")
    print(f"A sample chunk:\n {chunks[len(chunks)//2]}\n")

    # Step 3: Create vector DB and retriever (set save_vec_db=True to store it)
    embedder.save_vec_db = True  # Enable saving
    vector_db, retriever = embedder.create_vector_database(chunks)
    print("Vector database created and retriever is ready.")

    print("-"*20)