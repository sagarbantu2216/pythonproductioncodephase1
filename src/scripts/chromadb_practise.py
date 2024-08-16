from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

# Global variables for ChromaDB and embedding model
chromadb = None
embedding_model = None

def using_chromadb(all_text_chunks, docs):
    docs = docs
    print("3. Entered into using_chromaDB called function")
    global chromadb
    global embedding_model
    print("Entered into using_chromaDB called function")
    print(f"Number of text chunks created : {len(all_text_chunks)}")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = OpenAIEmbeddings(api_key=openai_api_key)
    # print(f"Embedding Model : {embedding_model}")
    
    chromadb = Chroma.from_documents(
        documents= all_text_chunks,
        embedding=embedding_model, 
        persist_directory="chromadb_persist_dir",
        collection_name="mycollection"
    )
    collection = chromadb.get() 
    # print(f"Collection : {collection}")
    # print(f"Collection name : {collection['name']}")
    print(f"Number of Collection : {len(collection['ids'])}")
    print(f"Total Collection ids in my Collection : {collection['ids']}")
    # from chain import retrieval_chain
    # retrieval_chain(chromadb, embedding_model, docs)
    return chromadb, embedding_model, docs
   
    
    