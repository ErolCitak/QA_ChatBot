import gradio as gr
import torch

import utils
import config
from llm import llm_wrapper
from embedding import embedder
from chain import rag_chain

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

class RAGApp:
    def __init__(self):
        # Device Setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model and Embedder Initialization
        self.embedder_name = config.EMBED_MODEL
        self.data_dir = config.DATA_DIR
        self.documents = config.DOCUMENTS

        self.my_embedder = embedder.Embedder(
            self.embedder_name, 
            self.data_dir, 
            self.documents, 
            chunk_length=512, 
            overlap=32, 
            save_vec_db=True
        )

        # Initialize LLM
        base_model_id = config.LLM_MODEL_ID
        self.my_llm_wrapper = llm_wrapper.SmolLLMWrapper(
            base_model_id, 
            max_length=256,
            temperature=0.3, 
            top_p=0.9, 
            top_k=50,
            repetition_penalty=1.2, 
            do_sample=True, 
            truncation=True
        )

        llm, llm_tokenizer = self.my_llm_wrapper.get_llm()

        self.pipe = pipeline(
            "text-generation",
            model=llm,
            tokenizer=llm_tokenizer,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True,
            truncation=True
        )

        self.llm_hg_pipeline = HuggingFacePipeline(pipeline=self.pipe)

        # Placeholders for retriever and chain
        self.retriever = None
        self.qa_bot_chain = None

    def load_document(self, doc_index):
        loaded_doc = self.my_embedder.pdf_data_loader(doc_index)
        cleaned_docs = utils.clean_business_conduct_policy(
            loaded_doc,
            n_remove_first_lines=3,
            n_discard_pages=[1, 2, 20]
        )
        chunks = self.my_embedder.text_splitter(cleaned_docs)
        vector_db, self.retriever = self.my_embedder.create_vector_database(
            chunks, 
            top_k_doc=5
        )

        rag_chain_train = rag_chain.RAGChainBuilder(
            self.llm_hg_pipeline, 
            self.retriever
        )
        self.qa_bot_chain = rag_chain_train.build_chain()

        return "✅ Document loaded and ready. You can now ask your question."

    def ask_question(self, user_question):
        if self.qa_bot_chain is None:
            return "❌ Please load a document first!"
        response = self.qa_bot_chain.run(user_question)
        return response

    def launch_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# 📚 RAG Question Answering Bot")

            with gr.Row():
                doc_selector = gr.Dropdown(
                    choices=[f"{i}: {doc}" for i, doc in enumerate(self.documents)],
                    label="Select a Document",
                    interactive=True
                )
                load_btn = gr.Button("Load Document")

            load_status = gr.Label("Please select and load a document.")

            with gr.Row():
                user_input = gr.Textbox(
                    lines=2, 
                    placeholder="Ask your question here...", 
                    label="Your Question"
                )
                ask_btn = gr.Button("Ask!")

            response_output = gr.Textbox(
                lines=10, 
                label="Response", 
                interactive=False
            )

            # Button Actions
            load_btn.click(
                fn=lambda x: self.load_document(int(x.split(":")[0])),
                inputs=doc_selector,
                outputs=load_status
            )

            ask_btn.click(
                fn=self.ask_question,
                inputs=user_input,
                outputs=response_output
            )

        demo.launch()


if __name__ == "__main__":
    app = RAGApp()
    app.launch_ui()