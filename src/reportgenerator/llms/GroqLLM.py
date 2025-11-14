import os
from langchain_groq import ChatGroq

def get_llm():
    
    # llm = ChatGroq(model="qwen/qwen3-32b")
    llm = ChatGroq(model="openai/gpt-oss-20b")
    return llm