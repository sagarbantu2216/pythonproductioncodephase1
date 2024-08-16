
#------------------- Create Stuff Document Chain ----------------------------------------
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os

def retrieval_chain(chromadb, embedding_model, docs):
    print(f"Number of documents passed as parameter: {len(docs)}")
    print("4. Entered into retrievalChain called function")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model = "gpt-3.5-turbo", api_key = openai_api_key)
    # question = "What is the dob of the Patient?"
    #------------------------- using Single Query Retriever by embedding question ------------------------------
    # query_embedding = embedding_model.embed_query(question)
    # retriever = chromadb.similarity_search_by_vector(query_embedding, k=1)
    # print(f"Number of documents retrieved : {len(retriever)}")
    # print(f"Retriever: {retriever}")
    # for doc in docs:     # Verifying document matches with the retriever
    #     doc_id = doc.metadata['unique_id']
    #     sess_id = doc.metadata['session_id']
    #     print(doc_id)
    #     print(sess_id)
    #     if doc_id == retriever[0].metadata['unique_id'] and sess_id == retriever[0].metadata['session_id']:
    #         print("Document found")
    #         print(f"Retriever: {retriever}")
    #         break
    # else:
    #     print("Document not found")
    #     return
    # context = "\n".join([doc.page_content for doc in retriever])
    # print(f"Retriever Context : {context}")

    #------------------------- usig Multi Query Retriever ------------------------------
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    retriever = chromadb.as_retriever()
    # system_prompt = (
    #     "You are an assistant for question-answering tasks. "
    #     "Use the following pieces of retrieved context to answer "
    #     "the question. If you don't know the answer, say that you "
    #     "don't know. Use three sentences maximum and keep the "
    #     "answer concise."
    #     "\n\n"
    #     "{context}"
    # )
    system_prompt = (
        "You are a medical assistant AI with access to a patient's medical records."
        "Your role is to provide detailed and accurate responses based on the patient's medical records."
        "Answer the questions based on the provided context only."
        "Please provide the most accurate response based on the question."
        "If you do not have an answer from the provided information say so."
        "Input will be in JSON format, and the output keys will be the same as the input keys"
        "The values of the input keys must be retrieved from the input text."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    questions =  {
        "age": "age of the patient",
        "dob": "dob of the patient",
        "gender": "gender of the patient",
        "race": "race of the patient",
        "ethnicity": "ethnicity of the patient",
        "smoking Status": "smoking status of the patient"
    }
    # Iterate over each question and invoke the retrieval chain
    responses = {}
    for key, question in questions.items():
        response = rag_chain.invoke({"input": question})
        responses[key] = response
    # response = rag_chain.invoke({"input" :questions})
    print(responses)
    answers = "\n".join([response['answer'] for response in responses.values()])
    print(f"Answers: {answers}")
    # print(response['answer'])
    # context = "\n".join([doc.page_content for doc in response['context']])
    # print(f"Retriever Context : {context}")
    
    # for doc in docs:
    #     doc_id = doc.metadata['unique_id']
    #     sess_id = doc.metadata['session_id']
    #     print(f"Checking Doc ID: {doc_id}, Session ID: {sess_id}")

    #     # Check each context in the response
    #     for context in response['context']:
    #         context_id = context.metadata['unique_id']
    #         context_sess_id = context.metadata['session_id']

    #         if doc_id == context_id and sess_id == context_sess_id:
    #             print("Document found")
    #             # print(f"Retriever: {doc}")
    #             break
    #     else:
    #         continue
    #     break
    # else:
    #     print("Document not found")
    
        

    #----------------------- Prompt Template Example code and invoking the chain ------------------------------
    # message = f"""
    # Answer this question using the provided context only.
    # {question}
    # Context:
    # {context}
    # """
    # print(f"Message: {message}")

    # prompt = ChatPromptTemplate.from_messages([("human", message)]) #creating prompt using ChatPromptTemplate
    # print(f"Prompt: {prompt}")

    # chain = {"context": RunnablePassthrough(), "question": RunnablePassthrough()} | prompt | llm  #creating chain using RunnablePassthrough
    # print(f"Chain : {chain}")

    # input_data = {"context": context, "question": question} #preparing input data for the chain

    # response = chain.invoke(input_data) #invoking the chain to get the response using input data
    # print(f"Response from LLM : {response}")
    # print(f"Response Content from LLM response : {response.content}") #printing the response content only

    print("5. End of retrievalChain called function")
    print("Deletion Process of Chroma DB Collection Started")
    chromadb.delete_collection() #deleting the collection after the retrieval process
    print("Deletion Process of Chroma DB Collection Completed")


# @app.route("/retrieval", methods=["POST"]) # API for retrieving the response for the question
# def retrieval():
#     try:
#         question = request.json["question"]
#         query_embedding = embedding_model.embed_query(question)
#         retriever = chromadb.similarity_search_by_vector(query_embedding, k=1)
#         response = []
#         for doc in retriever:
#             response.append({
#                 "metadata": doc.metadata,
#                 "page_content": doc.page_content
#             })
#         return jsonify(response), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(port=8000)