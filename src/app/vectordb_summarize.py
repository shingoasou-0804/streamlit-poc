import chromadb
import openai
import os
import streamlit as st
import wikipediaapi

from chromadb.utils import embedding_functions
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


embedding_model_name = "stsb-xlm-r-multilingual"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model_name
    )
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
        name="test",
        embedding_function=embedding_fn
)


def get_wikitext(wiki_title):
    wiki = wikipediaapi.Wikipedia("llm-wiki-search", "ja")
    wiki_file = os.path.join("./", f"__{wiki_title}.txt")
    if not os.path.exists(wiki_file):
        page = wiki.page(wiki_title)
        if not page.exists():
            return f"{wiki_title}が存在しません。"
        with open(wiki_file, "w", encoding="utf-8") as f:
            f.write(page.text)
    with open(wiki_file, "r", encoding="utf-8") as f:
        text = f.read()
        print(f"=== Wikipedia: {wiki_title} ({len(text)}字) ===")
        return text


def insert_text(text, chunk_size=500):
    chunks = []
    paragraphs = text.split("\n")
    cur = ""
    for s in paragraphs:
        cur += s + "\n"
        if len(cur) > chunk_size:
            chunks.append(cur)
            cur = ""
    if cur != "":
        chunks.append(cur)
    print(f"=== チャンク数: {len(chunks)} ===\n")
    collection.add(
        ids=[f"{text[0:5]}{i+1}" for i in range(len(chunks))],
        documents=chunks
    )


def query_text(query, max_len=1500):
    docs = collection.query(
        query_texts=[query],
        n_results=10,
        include=["documents"]
    )
    doc_list = docs["documents"][0]
    doc_result = ""
    for doc in doc_list:
        if len(doc_result + doc) > max_len:
            break
        doc_result += doc.strip() + "\n-----\n"
    return doc_result


def llm_summarize(text, query):
    insert_text(text)
    doc_result = query_text(query)
    prompt = \
        f"### 指示:\n以下の情報を参考にして要約してください。\n" + \
        f"特に「{query}」に注目して下さい。\n" + \
        f"### 情報:\n```{doc_result}```\n"
    print("=== 回答プロンプト ===\n" + prompt)
    result = call_chatgpt(prompt)
    print("=== 結果 ===\n" + result)
    return result


def call_chatgpt(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    return completion.choices[0].message.content
