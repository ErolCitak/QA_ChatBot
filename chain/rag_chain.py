import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch # type: ignore

import config
from langchain.schema import HumanMessage, SystemMessage, AIMessage # type: ignore
from langchain.chains import RetrievalQA # type: ignore
from langchain.prompts import PromptTemplate # type: ignore

class RAGChainBuilder:
    def __init__(self, llm, retriever):

        print("Initializing Retrieval Chain")
        print("-"*20)
        self.llm = llm
        self.retriever = retriever

    def get_prompt_template(self):
        template = """
        ### Context:
        {context}

        ### Instruction:
        You are a helpful assistant. Answer the question based on the above context provided.        

        ### Question:
        {question}

        ### Answer:
        """
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )        

    def build_chain(self):
        prompt = self.get_prompt_template()

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )