# Importing Necessary Libraries
from flask import Flask, request, jsonify
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from flask_cors import CORS
import os


app = Flask(__name__)  # Create a Flask app
CORS(app)

openai_api_key =  os.getenv("OPENAI_API_KEY")  # Get the OpenAI API key from the environment variables

global chromadb, embedding_model, docs
#------------------ API Endpoint for uploading documents and processing it ----------------------------------------------
@app.route("/upload_documents", methods=["POST"]) # API for uploading the documents
def upload_documents():
    # delete_collection() # Delete the collection before uploading the new documents
    print("Old collection deleted successfully!!")
    uploaded_files = request.files.getlist("files")
    # dir_path = request.form.get("directory_path")
    user_id = request.form.get("user_id")
    session_id = request.form.get("session_id")
    doc_id = request.form.get("doc_id")
    # print(f"Directory Path: {dir_path}")
    print(f"length of uploaded files: {len(uploaded_files)}")
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    print(f"Document ID: {doc_id}")
    from document_loader import text_document_loader
    global chromadb, embedding_model, docs
    chromadb, embedding_model, docs = text_document_loader(uploaded_files, user_id, session_id, doc_id) #Function call for text document loader
    return jsonify({"message": "Documents uploaded successfully"}), 200

#-----------------------API Endpoint for finding the similar documents for the query ----------------------------------------------
@app.route("/find_similar", methods=["POST"]) # API for finding similar documents for the Query 
def find_similar():
    try:
        query = request.json["query"]
        query_embedding = embedding_model.embed_query(query)
        result = chromadb.similarity_search_by_vector(query_embedding, k=3) # k is the number of similar documents to be returned
        print(len(result))
        response = []
        for doc in result:
            print(f"Document: {doc.metadata}")
            print(f"Page Content: {doc.page_content}")
            response.append({
                "metadata": doc.metadata,
                "page_content": doc.page_content
            })
        
        return jsonify(response),200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#-----------------------API Endpoint for getting the document by doc ID ----------------------------------------------
@app.route("/get_document/<doc_id>", methods=["GET"]) # API for getting the document by ID
def get_document(doc_id):
    try:
        doc = chromadb.get(doc_id)
        print(doc)
        if doc:
            response = {"metadata": doc['metadatas'], "page_content": doc['documents']}
            return jsonify(response), 200
        else:
            return jsonify({"message": "Document not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#-----------------------API Endpoint for getting response from LLM using user question ----------------------------------------------
@app.route("/ask_your_question", methods=["POST"]) # API for retrieving the response for the question
def ask_your_question():
    question = request.json.get('question')
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    # retriever = chromadb.as_retriever()
    retriever = chromadb.as_retriever(
        search_type="mmr",  # Example search type, adjust as needed
        search_kwargs={
            "filters": {
                "user_id": "001",
                "session_id": "session1"
            }
        }
    )
    system_prompt = (
        "You are a medical assistant AI with access to a patient's medical records."
        "Your role is to provide detailed and accurate responses based on the patient's medical records."
        "Answer the questions based on the provided context only."
        "Please provide the most accurate response based on the question."
        "If you do not have an answer from the provided information say so."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4-1106-preview", 
                     api_key=openai_api_key, 
                     temperature = 0, 
                     max_tokens=4000)
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": question})
    # print(response)
    # context = response.get("context", "")
    # print(type(context))
    # print(context)
    # for document in context:
    #     print(document.metadata)
    answer = response.get("answer", "")
    return jsonify({"response" : answer}), 200

#---------------------Using Custom Agents and Tools for nlp headers attributes -----------------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

def retrieve_header_attributes_answer(question: str) -> str:
    """Retrieve an answer from the ChromaDB based on the question."""
    print("Entered into retrieve_answer function")
    retriever1 = chromadb.as_retriever()
    system_prompt1 = (
        "You are a medical assistant AI with access to a patient's medical records."
        "Your role is to provide detailed and accurate responses based on the patient's medical records."
        "Answer the questions based on the provided context only."
        "Please provide the most accurate response based on the question."
        "If you do not have an answer from the provided information say so."
        "\n\n"
        "{context}"
    )
    prompt1 = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt1),
            ("human", "{input}"),
        ]
    )
    llm1 = ChatOpenAI(model="gpt-3.5-turbo", 
                     api_key=openai_api_key, 
                     temperature = 0, 
                     max_tokens=4000)
    question_answer_chain_header = create_stuff_documents_chain(llm1, prompt1)
    rag_chain_header = create_retrieval_chain(retriever1, question_answer_chain_header)
    response = rag_chain_header.invoke({"input": question})
    print(f"Response: {response}")
    return response["answer"]

class CustomAgentHeaderAttributes:
    def __init__(self, tools):
        self.tools = tools

    def invoke(self, input_data):
        print("Entered into Custom Agent invoke function")
        question = input_data.get("input", "")
        print(f"Question: {question}")
        for tool in self.tools:
            print("Type of tool: ", type(tool))
            response = tool(str(question))
            if response:
                return {"answer": response}
        return {"answer": "No answer found"}

tools = [retrieve_header_attributes_answer]
agentHeader = CustomAgentHeaderAttributes(tools)

@app.route("/header_attributes_response", methods=["POST"])
def retrieval_agent_headers_attributes():
    print("Entered into Retrieval Agent hit API is called")
    question_map = request.json.get('question')
    print(f"Questions : {question_map}")
    print("Invoking agent...")
    responses = {}
    for key, value in question_map.items():
        response = agentHeader.invoke({"input": value})
        answer = response.get("answer", "")
        responses[key] = answer
        # responses.setdefault("answer", []).append(answer)

    import json 
    import re
    result = {} # initialize an empty dictionary to store the final result
    # process each item in the answers list
    for key, answer in responses.items():
        try:
            clean_item = re.sub(r'```json|```', '', answer).strip()
            result[key] = clean_item
            # parsed_item = json.loads(clean_item)
            # result.update(parsed_item)
        except json.decoder.JSONDecodeError:
            print(f"Error parsing JSON: {answer}")
            result[key] = answer  # Store a part of the non-JSON response as a key

    # return json.dumps(result, indent=2), 200.
    return jsonify(result), 200

#-----------Using Custom Agents and Tools for getting the section response from llm in list format ---------------------------------------------------------
def retrieve_section_answers(question: str) -> str:
    """Retrieve an answer from the ChromaDB based on the question."""
    print("Entered into retrieve_answer_res function")
    retriever2 = chromadb.as_retriever()
    system_prompt2 = (
        "You are a medical assistant AI with access to a patient's medical records."
        "Your role is to provide detailed and accurate responses based on the patient's medical records."
        "Answer the questions based on the provided context only."
        "Please provide the most accurate response based on the question."
        "If you do not have an answer from the provided information, say so."
        "Ensure that all responses are provided as lists, even if there's only one item."
        "Input will be in JSON format, and the output keys will be the same as the input keys."
        "The values of the input keys must be retrieved from the input text."
        "\n\n"
        "{context}"
    )
    prompt2 = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt2),
            ("human", "{input}"),
        ]
    )
    llm2 = ChatOpenAI(model="gpt-4-1106-preview", api_key=openai_api_key, temperature = 0, max_tokens=4000)
    question_answer_chain = create_stuff_documents_chain(llm2, prompt2)
    rag_chain = create_retrieval_chain(retriever2, question_answer_chain)
    response = rag_chain.invoke({"input": question})
    # print(f"Response: {response}")
    return response["answer"]

class RetrievalAgentSectionAnswers:
    def __init__(self, tools):
        self.tools = tools

    def invoke(self, input_data):
        print("Entered into Custom Agent invoke function")
        question = input_data.get("input", "")
        print(f"Question: {question}")
        for tool in self.tools:
            print("Type of tool: ", type(tool))
            response = tool(str(question))
            if response:
                return {"answer": response}
        return {"answer": "No answer found"}

retrieval_tools = [retrieve_section_answers]
retrieval_agent = RetrievalAgentSectionAnswers(retrieval_tools)

@app.route("/retrieval_results_custom_agent_tool", methods=["POST"])
def retrieval_agent_section_answers():
    print("Entered into Retrieval Agent hit API is called")
    question = request.json.get('question')
    print(f"Type of question: type({question})")
    print("Invoking agent...")
    responses = {}
    for value in question:
        response = retrieval_agent.invoke({"input": value})
        answer = response.get("answer", "")
        responses.setdefault("answer", []).append(answer)
    for key, value in responses.items():
        print(f"Key: {key}, Value: {value}")

    import json 
    import re
    result = {} # initialize an empty dictionary to store the final result
    # process each item in the answers list
    for item in responses["answer"]:
        try:
            clean_item = re.sub(r'```json|```', '', item).strip()
            parsed_item = json.loads(clean_item)
            result.update(parsed_item)
        except json.decoder.JSONDecodeError:
            print(f"Error parsing JSON: {item}")
    print("End of Retrieval Agent hit API is called")
    print(f"Result: {result}")      #printing the result dictionary 
    result_sections_dict = json.dumps(result, indent=2)
    process_sections_result(result_sections_dict)
    return json.dumps(result, indent=2), 200

#-------------- Processing the sections result into one list to create the nlp response schema ----------------------------------------------
# @app.route("/process_sections_result", methods=["POST"])
def process_sections_result(result_sections_dict):
    import json
    print(result_sections_dict)
    print(f"result_sections_dict type: {type(result_sections_dict)}")
    print(f"result_sections_dict value: {result_sections_dict}")
    if isinstance(result_sections_dict, str):
        try:
            # Attempt to parse it if it's a JSON string
            result_sections_dict = json.loads(result_sections_dict)
        except json.JSONDecodeError:
            return jsonify({"error": "result_sections_dict is not a valid JSON string"}), 400
    combined_list = result_sections_dict.get("Problem List", [])  # Initialize the combined list with the problem list

    for key, value in result_sections_dict.items():
        if key == "Problem List":
            continue  # Skip "Problem List" because it's already addedc

        if isinstance(value, list):
            # If the value is a list, extend the combined list with this list
            combined_list.extend(value)
        elif isinstance(value, str):
            # If the value is a string, format it with the key and add to the combined list
            combined_list.append(f"{key.replace('_', ' ')}: {value}")
        else:
            # If the value is of an unexpected type, you can handle it here (optional)
            print(f"Unexpected value type for key '{key}': {type(value)}")
    global final_result
    final_result = {  # Create the final JSON object with one key
        "problems": combined_list
    }
    final_result_json = json.dumps(final_result, indent=2)  # Convert the final result to JSON format
    print(final_result_json)
    
    format_medical_problems() # Call the API endpoint to format the medical problems NLP structure

#------------------------- Using custom tool for schema arrangement for the llm response ----------------------------------------------
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def generate_prompt(problem: str) -> str:
    print("4. Entered into generate_prompt function")
    print(f"Problem Received: {problem}")
    return f"""
    Given the medical problem: "{problem}", generate a JSON schema that includes the following attributes:
    - `name`: Based on the type of problem (e.g., DiseaseDisorderMention, LabMention, etc.)
    - `sectionName`: A relevant section name such as "History of Present Illness" or "Past Medical History".
    - `sectionOid`: A unique OID for the section, such as "2.16.840.1.113883.10.20.22.2.20".
    - `sectionOffset`: A simulated character offset range for the section.
    - `sentence`: A simulated character offset range for the sentence.
    - `extendedSentence`: A simulated character offset range for the extended sentence.
    - `text`: The text of the problem with simulated character offsets.
    - `attributes`: A set of attributes such as `derivedGeneric`, `polarity`, `relTime`, `date`, `status`, etc., with dynamically generated values relevant to the problem.
    - `umlsConcept`: A list of concepts with attributes like `codingScheme`, `cui`, `tui`, `code`, and `preferredText` that relate to the problem.

    Use the structure of the following JSON schema as an example and fill in the values accordingly:
    {{
        "name": "'DiseaseDisorderMention', 'LabMention', 'MedicationMention', etc.",  // Type of mention
        "sectionName": "appropriate section name, omit if 'SIMPLE_SEGMENT'",  // Generate section name based on the problem
        "sectionOid": "appropriate OID or 'SIMPLE_SEGMENT' if not applicable",  // Generate section OID based on the problem section name
        "sectionOffset": [start_offset, end_offset],  // Character offset range for the section
        "sentence": [start_offset, end_offset],  // Character offset range for the sentence
        "extendedSentence": [start_offset, end_offset],  // Extended offset range
        "text": ["{problem}", start_offset, end_offset],  // Problem text
        "attributes": {{
            "derivedGeneric": "1 or 0",  // Indicates if the term is generic
            "polarity": "positive or negated",  // Polarity of the mention
            "relTime": "current status, history status",  // Time relation of the problem
            "date": "MM-DD-YYYY",  // Date associated with the problem
            "status": "stable, unstable",  // Status of the problem
            "medDosage": "medication dosage if applicable",  // Medication dosage
            "medForm": "medication form if applicable",  // Medication form
            "medFrequencyNumber": "frequency number if applicable",  // Medication frequency number
            "medFrequencyUnit": "frequency unit if applicable",  // Medication frequency unit
            "medRoute": "medication route if applicable",  // Medication route
            "medStrengthNum": "strength number if applicable",  // Medication strength number
            "medStrengthUnit": "strength unit if applicable",  // Medication strength unit
            "labUnit": "lab unit if applicable",  // Lab unit
            "labValue": "lab value if applicable",  // Lab value
            "umlsConcept": [
                {{
                    "codingScheme": "ICD10CM or RxNorm ",  // Coding scheme
                    "cui": "CUI code based on problem",  // Generate UMLS CUI code
                    "tui": "TUI code based on problem",  // Generate UMLS TUI code
                    "code": "ICD10CM code or RxNorm code generated from the above codingSchema",  // Generate Relevant medical code
                    "preferredText": "Get ICD10 code description and RxNorm code description"  // Get the description based on the code
                }}
            ]
        }}
    }}
    """
 
def format_problem_with_schema(problem: str) -> dict:
    print("3. Entered into format_problem_with_schema function")
    prompt = generate_prompt(problem)
    print("Finished generating prompt")
    system_prompt3 = (
        "You are a medical assistant AI with access to a patient's medical records."
        "Your role is to provide detailed and accurate responses based on the patient's medical records."
        "Answer the questions based on the provided context only."
        "Please provide the most accurate response based on the question."
        "If you do not have an answer from the provided information, say so."
        "Ensure that all responses are provided as lists, even if there's only one item."
        "Input will be in JSON format, and the output keys will be the same as the input keys."
        "Retrieve the values of the input keys directly from the provided context."
        "If the problem mentioned is a disease or condition, use the ICD10CM coding schema."
        "If the problem mentioned is a medication, use the RxNorm coding schema."
        "\n\n"
        "{context}"
    )
    # Preparing the message for the chat-based model
    chat_messages = [
        {"role": "system", "content": system_prompt3},
        {"role": "user", "content": prompt}
    ]
    llm3 = ChatOpenAI(model="gpt-4o",
                      api_key=openai_api_key,
                      temperature=0,
                      max_tokens=4000)  
    response = llm3.invoke(chat_messages)
    return response
 
class CustomAgentWithSchema:
    print("Entered into CustomAgentWithSchema class")
    def __init__(self, tools):
        self.tools = tools

    def invoke(self, input_data):
        print("2. Entered into CustomAgentWithSchema invoke function")
        problem = input_data.get("input", "")
        print(f"Problem: {problem}")
        for tool in self.tools:
            response = tool(problem)
            if response:
                return {"answer": response}
        return {"answer": "No answer found"}
 
tools_schema = [format_problem_with_schema]
agent_schema = CustomAgentWithSchema(tools_schema)
 
@app.route("/format_medical_problems", methods=["POST"])
def format_medical_problems():
    print(" 1.Entered into format_medical_problems hit API is called")
    import json
    problems = final_result.get("problems", [])
    print(f"Problems: {problems}")
    print(f"Problems: {problems}")
    responses = []
    for problem in problems:
        response = agent_schema.invoke({"input": problem})
        # print(f"Response: {response}")
        responses.append(response.get("answer", ""))

    for response in responses:
        content = response.content
        cleaned_response = content.replace("```json", "").replace("```", "").strip()
        print(cleaned_response)
    print("End of format_medical_problems hit API is called")
    print("NLP response schema created successfully!!")
    
    return jsonify({responses}), 200

#------------- API Endpoint for deleting the collection inside Chroma Persist directory ----------------------------------------------
@app.route("/delete_collection", methods=["POST"]) # API for deleting the collection -- working
def delete_collection():
    try:
        chromadb.delete_collection()
        return "Collection deleted successfully from ChromaDB!!", 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#------------- API Endpoint for deleting all the collections inside Chroma Persist directory ----------------------------------------------
@app.route("/delete_all_collections", methods=["POST"]) # API for deleting all the collections -- working
def delete_all_collections():
    try:
        collections = chromadb.list_collections()
        print(f"Number of collections: {len(collections)}")
        for collection in collections:
            chromadb.delete_collection(collection)
            print(f"Collection {collection} deleted successfully!!" )
        return "All collections deleted successfully from ChromaDB!!", 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#---------------------- New code for nlp response umls concepts ------------------------------

# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# # Define your system prompt
# system_prompt2 = (
#     "You are a medical assistant AI with access to a patient's medical records."
#     "Your role is to provide detailed and accurate responses based on the patient's medical records."
#     "Answer the questions based on the provided context only."
#     "Please provide the most accurate response based on the question."
#     "If you do not have an answer from the provided information, say so."
#     "Ensure that all responses are provided as lists, even if there's only one item."
#     "Input will be in JSON format, and the output keys will be the same as the input keys."
#     "The values of the input keys must be retrieved from the input text."
#     "\n\n"
#     "{context}"
# )
# def generate_prompt(problems: list) -> str:
#     print("4. Entered into generate_prompt function")
#     problems_str = "\n".join([f"- {problem}" for problem in problems])
#     return f"""
#     Given the following medical problems: 
#     {problems_str}
#     Generate a JSON schema for each problem that includes the following attributes:
#     - `name`: Based on the type of problem (e.g., DiseaseDisorderMention, LabMention, etc.)
#     - `sectionName`: A relevant section name such as "History of Present Illness" or "Past Medical History".
#     - `sectionOid`: A unique OID for the section, such as "2.16.840.1.113883.10.20.22.2.20".
#     - `sectionOffset`: A simulated character offset range for the section.
#     - `sentence`: A simulated character offset range for the sentence.
#     - `extendedSentence`: A simulated character offset range for the extended sentence.
#     - `text`: The text of the problem with simulated character offsets.
#     - `attributes`: A set of attributes such as `derivedGeneric`, `polarity`, `relTime`, `date`, `status`, etc., with dynamically generated values relevant to the problem.
#     - `umlsConcept`: A list of concepts with attributes like `codingScheme`, `cui`, `tui`, `code`, and `preferredText` that relate to the problem.

#     Use the structure of the following JSON schema as an example and fill in the values accordingly:
#     {{
#         "name": "'DiseaseDisorderMention', 'LabMention', 'MedicationMention', etc.",  // Type of mention
#         "sectionName": "appropriate section name, omit if 'SIMPLE_SEGMENT'",  // Generate section name based on the problem
#         "sectionOid": "appropriate OID or 'SIMPLE_SEGMENT' if not applicable",  // Generate section OID based on the problem section name
#         "sectionOffset": [start_offset, end_offset],  // Character offset range for the section
#         "sentence": [start_offset, end_offset],  // Character offset range for the sentence
#         "extendedSentence": [start_offset, end_offset],  // Extended offset range
#         "text": ["{problems_str}", start_offset, end_offset],  // Problem text with character offsets
#         "attributes": {{
#             "derivedGeneric": "1 or 0",  // Indicates if the term is generic
#             "polarity": "positive or negated",  // Polarity of the mention
#             "relTime": "current status, history status",  // Time relation of the problem
#             "date": "MM-DD-YYYY",  // Date associated with the problem
#             "status": "stable, unstable",  // Status of the problem
#             "medDosage": "medication dosage if applicable",  // Medication dosage
#             "medForm": "medication form if applicable",  // Medication form
#             "medFrequencyNumber": "frequency number if applicable",  // Medication frequency number
#             "medFrequencyUnit": "frequency unit if applicable",  // Medication frequency unit
#             "medRoute": "medication route if applicable",  // Medication route
#             "medStrengthNum": "strength number if applicable",  // Medication strength number
#             "medStrengthUnit": "strength unit if applicable",  // Medication strength unit
#             "labUnit": "lab unit if applicable",  // Lab unit
#             "labValue": "lab value if applicable",  // Lab value
#             "umlsConcept": [
#                 {{
#                     "codingScheme": "ICD10CM or SNOMEDCT_US",  // Coding scheme
#                     "cui": "CUI code based on problem",  // Generate UMLS CUI code
#                     "tui": "TUI code based on problem",  // Generate UMLS TUI code
#                     "code": "ICD10CM or SNOMEDCT_US code generated from the above codingSchema",  // Generate Relevant medical code
#                     "preferredText": "Get ICD10 code description or SNOMED description"  // Get the description based on the code
#                 }}
#             ]
#         }}
#     }}
#     """

# def format_problem_with_schema(problems: list) -> list:
#     print("3. Entered into format_problem_with_schema function")
#     prompt = generate_prompt(problems)
#     print("Finished generating prompt")
    
#     chat_messages = [
#         {"role": "system", "content": system_prompt2},
#         {"role": "user", "content": prompt}
#     ]
    
#     # Assuming `llm` is your instance of ChatOpenAI
#     llm = ChatOpenAI(model="gpt-4o",
#                      api_key=openai_api_key,
#                      temperature = 0.1,
#                      max_tokens = 4000)  # Replace with your model
#     response = llm.invoke(chat_messages)
    
#     return response

# class CustomAgentWithSchema:
#     print("Entered into CustomAgentWithSchema class")
#     def __init__(self, tools):
#         self.tools = tools

#     def invoke(self, input_data):
#         print("2. Entered into CustomAgentWithSchema invoke function")
#         problems = input_data.get("input", [])
#         print(f"Problems: {problems}")
#         for tool in self.tools:
#             response = tool(problems)
#             if response:
#                 return {"answer": response}
#         return {"answer": "No answer found"}

# tools_schema = [format_problem_with_schema]
# agent_schema = CustomAgentWithSchema(tools_schema)

# @app.route("/format_medical_problems_latest", methods=["POST"])
# def format_medical_problems_latest():
#     import json
#     print(" 1.Entered into format_medical_problems hit API is called")
#     problems = request.json.get('problems')
#     print(f"Problems: {problems}")
    
#     response = agent_schema.invoke({"input": problems})
#     content = response.get('answer').content
#     content = content.strip('```json\n').strip('```')
#     print(content)
#     # # Parse the JSON content
#     # try:
#     #     json_content = json.loads(content)
#     #     print(json_content)
#     # except json.JSONDecodeError as e:
#     #     print(f"Failed to decode JSON: {e}")
#     #     return jsonify({"error": "Failed to decode JSON"}), 500
    
#     return jsonify({"message": "Success"}), 200

#----------------------  Main function to run the app ----------------------------------------------
if __name__ == "__main__":
    # from document_loader import text_document_loader
    # chromadb, embedding_model, docs = text_document_loader()
    # print("Got Chromadb, Embedding model and docs in return to Chroma DB Flask")
    app.run(host='0.0.0.0', port=9000) # Run the app on port 5000 for chromadb analysis and retrieval as well
