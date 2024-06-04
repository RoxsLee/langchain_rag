# -*- coding: UTF-8 -*-
"""
@Project : langchain_rag 
@File    : langchain_rag_demo_0.py
@Author  : lixianbo
@Date    : 2024/6/2 17:21

https://www.cnblogs.com/windpoplar/articles/18046618
"""
from langchain.indexes import VectorstoreIndexCreator
from pprint import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from chatbot import llm, deepseek_api_key

llm = ChatOpenAI(openai_api_key=deepseek_api_key,
                 openai_api_base="https://api.deepseek.com",
                 model_name="deepseek-chat")

# 加载文档
loader = CSVLoader(file_path='ordersample.csv')
data = loader.load()

# 标准化输出
pprint(data)

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(data)

# 向量化存储
model_name = "BAAI/bge-large-zh-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
bgeEmbeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
vector = FAISS.from_documents(all_splits, bgeEmbeddings)

# 向量化检索
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# out = retriever.invoke("收货人姓名是张三丰的，有几个订单？金额分别是多少，总共是多少？")

# 生成问答结果
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""仅根据所提供的上下文回答以下问题:

<context>
{context}
</context>

问题: {question}""")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
retriever_chain = (
    {"context": retriever , "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

out = retriever_chain.invoke("订单ID是123456的收货人是谁，电话是多少?")
print(out)
