# -*- coding: UTF-8 -*-
"""
@Project : langchain_rag 
@File    : chatbot.py
@Author  : lixianbo
@Date    : 2024/6/2 18:10 
"""
# python3
# Please install OpenAI SDK first：`pip3 install openai`
from openai import OpenAI

deepseek_api_key = "sk-207dbb7fe4db4a70a00e4a1fb15b39dc"
llm = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")


def chat_deepseek(chat_text):
    response = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": chat_text},
        ],
        stream=False
    )

    return response.choices[0].message.content