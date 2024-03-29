import torch
from transformers import BitsAndBytesConfig
from llama_index import ServiceContext
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM

embed_model = "local:BAAI/bge-small-en-v1.5"

def load_llm():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    llm = HuggingFaceLLM(
        model_name="HuggingFaceH4/zephyr-7b-beta",
        tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
        query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
        context_window=3900,
        max_new_tokens=256,
        model_kwargs={"quantization_config": quantization_config},
        # tokenizer_kwargs={},
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        messages_to_prompt=messages_to_prompt,
        device_map="auto",
    )
    
    return llm
    
    
def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == 'system':
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == 'assistant':
            prompt += f"<|assistant|>\n{message.content}</s>\n"
            
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt
        
    prompt = prompt + "<|assistant|>\n"
    
    return prompt


    
