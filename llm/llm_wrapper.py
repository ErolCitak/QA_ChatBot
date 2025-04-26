import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
import torch # type: ignore
from torch import compile # type: ignore
from peft import PeftModel # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # type: ignore
from langchain_huggingface import HuggingFacePipeline # type: ignore
from langchain_core.runnables import Runnable # type: ignore
from langchain.schema import HumanMessage, SystemMessage, AIMessage # type: ignore

import warnings
warnings.filterwarnings('ignore')


class SmolLLMWrapper:
    def __init__(self, base_model_id, max_length=512, temperature=0.3, top_p=0.9, top_k=50,
                         repetition_penalty=1.2, do_sample=True, truncation=True):
        
        print("Initializing LLM Model")
        print("-"*20)

        # Some hyper-param. Initialization
        self.my_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.truncation = truncation


        # Base model initialization
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left", padding="max_length", max_length=self.max_length)
        self.base_tokenizer.pad_token = self.base_tokenizer.eos_token or self.base_tokenizer.unk_token
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map=self.my_device)

    def get_llm(self):
        return self.base_model, self.base_tokenizer


    #{"role": "user", "content": "Who are you? Please, answer in pirate-speak."}
    def llm_template(self, user_input_message, include_system=True):
        # messages = [{"role": "user", "content": "What is the capital of France."}]

        # 1. Create LangChain-style messages
        langchain_messages = []

        if include_system:
            langchain_messages.append(SystemMessage(content="You are a helpful assistant."))

        langchain_messages.append(HumanMessage(content=user_input_message))

        # 2. Convert to OpenAI-style message dicts
        role_map = {
            HumanMessage: "user",
            AIMessage: "assistant",
            SystemMessage: "system"
        }

        messages = [
            {"role": role_map[type(m)], "content": m.content}
            for m in langchain_messages
        ]

        return messages

    def llm_generate_output(self, message):
        
        # Apply tokenization
        input_text=self.base_tokenizer.apply_chat_template(message, tokenize=False)
        
        # Encode input
        input = self.base_tokenizer.encode(input_text, return_tensors="pt").to(self.my_device)
        
        # Generate & Decode output
        output = self.base_model.generate(input, max_new_tokens=self.max_length, temperature=self.temperature,
                                     top_p=self.top_p, top_k=self.top_k, do_sample=self.do_sample)
        
        output = self.base_tokenizer.decode(output[0], skip_special_tokens=True)

        return output


if __name__ == "__main__":

    our_base_model_id = config.LLM_MODEL_ID

    smollm = SmolLLMWrapper(our_base_model_id)
    user_message = smollm.llm_template(user_input_message="Who are you?", include_system=True)
    model_output = smollm.llm_generate_output(user_message)

    print(f"User Input:\n{user_message}")
    print(f"Model Output:\n{model_output}")