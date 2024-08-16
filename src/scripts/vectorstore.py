import os
def vectorStoreDB(docs):
    print("Entered into Vector Store DB called function")
    print(f"Number of documents: {len(docs)}")
    #--------------------------- Emnedding model ---------------------------------
    from langchain_openai import OpenAIEmbeddings
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = OpenAIEmbeddings(api_key=openai_api_key)
    # print(embedding_model)
    #--------------------------- Vector Store using Simple Chromadb ---------------------------------
    from langchain_chroma import Chroma
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model)
    print(vectorstore)
    #-----------------Similarity search ---------------------------------------
    # query = "What is the patient's age and date of birth?"
    # result = vectorstore.similarity_search(query)
    # print(result[0].page_content)
    #------------------ Similarity search by vector ------------------------------
    # query = "What is the patient's age and date of birth?"
    # query_vector = embedding_model.embed_query(query)
    # result1 = vectorstore.similarity_search_by_vector(query_vector)
    # print(result1)
    #--------------------------------- Vector Store using Chromadb ---------------------------------
    # import chromadb
    # from chromadb.config import Settings
    # settings = Settings(persist_directory="chromadb_persist_dir")
    # chroma_client = chromadb.Client(settings)
    # print(f"Chroma DB Client : {chroma_client}")
    # collection_name = "collection-1"
    # collection = chroma_client.get_or_create_collection(name = collection_name)
    # print(f"Collection Name: {collection.name}")
    # texts = [doc.page_content for doc in docs]
    # print(f"Number of Texts : {len(texts)}")
    # # print(texts)
    # embeddings = embedding_model.embed_documents(texts)
    # print(f"Generated Embeddings: {embeddings}")
    # print(f"Number of Embeddings : {len(embeddings)}")
    # for doc, embedding in zip(docs, embeddings):
    #     collection.add(ids = [doc.metadata['unique_id']], documents = [doc.page_content], embeddings = [embedding])
    # print(f"Number of documents in the collection after addition: {collection.count()}")
    # print(f"Collection name: {collection.name}")
    # fetched_docs = collection.get(ids=[doc.metadata['unique_id'] for doc in docs])
    # print(f"Fetched Documents: {fetched_docs}")
    # print(f"Number of Fetched Docs : {len(fetched_docs)}")
    # def query_embedding(query):
    #     query_embedding = embedding_model.embed_documents([query])[0]
    #     results = collection.query(query_embeddings = [query_embedding], n_results = 3)
    #     return results
    # query = "what are the medications taken by the patient?"
    # query_embedding = query_embedding(query)
    # print(query_embedding)
    # print("Query Results:")
    # for result in query_embedding['documents']:
    #     print(result)
    # print("Starting the deletion process of the collection")
    # chroma_client.delete_collection(collection_name)
    # # Verify the collection deletion
    # try:
    #     print(f"Collection count after deletion: {collection.count()}")
    # except Exception as e:
    #     print(f"Collection deleted, error occurred when trying to access it: {e}")

    #--------------------------- Vector Store using Pinecone ---------------------------------
    # from pinecone import Pinecone
    # pc = Pinecone(api_key= pinecone_api_key) # Initialize Pinecone
    # print(pc)
    # pinecone_index_name = "index-1" # Name of the index in Pinecone
    # index = pc.Index(pinecone_index_name) # Create an index in Pinecone
    # print(index)
    # vectors_to_upsert = []
    # for doc in docs:
    #     embeddings = embedding_model.embed_documents(doc.page_content)
    #     print(embeddings)
    #     vectors_to_upsert.append(embeddings)
    # print(len(vectors_to_upsert))
    # index.upsert(vectors_to_upsert) # Insert the embeddings into Pinecone index

