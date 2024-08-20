from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from langchain_chroma import Chroma
import chardet


# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Initialize the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Approx 1000 characters
    chunk_overlap=100,  # Small overlap to maintain context between chunks
)

def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File {file_path} deleted successfully.")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

@app.route("/upload", methods=["POST"])
def upload_files():
    if 'files' not in request.files:
        return "No files part in the request", 400
    
    files = request.files.getlist('files')
    user_id = request.form.get('userId')
    upload_id = request.form.get('uploadId')

    if not user_id or not upload_id:
        return "Missing userId or uploadId in the request body.", 400

    responses = []

    try:
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            text_content = ""

            if file.mimetype == 'application/pdf':
                file_data = PyPDFLoader(file_path)
                pdf_data = file_data.load_and_split()
                text_content = pdf_data[0].page_content
                print("PDF file uploaded")
            elif file.mimetype == 'text/plain':
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding']
                text_loader = TextLoader(file_path, encoding=encoding).load()
                text_content = text_loader[0].page_content
            elif file.mimetype == 'text/csv':
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding']
                file_data = CSVLoader(file_path, encoding=encoding).load()
                text_content = file_data[0].page_content
            else:
                return "Unsupported file type", 400

            if not text_content:
                print("No text content found in the file")
                continue

            openai_embeddings = OpenAIEmbeddings(
                api_key=os.getenv('OPENAI_API_KEY'))

            document = Document(page_content=text_content, metadata={'userId': user_id, 'uploadId': upload_id})
            split_docs = splitter.split_documents([document])

            vector_store = Chroma.from_documents(split_docs, openai_embeddings)
            app.vector_store = vector_store

            responses.append({'filename': filename})

            # Delete the file after processing
            delete_file(file_path)

        return jsonify(responses), 200

    except Exception as e:
        print(f"Error processing files: {e}")
        return f"Error processing files: {e}", 500


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    user_id = data.get('userId')
    upload_id = data.get('uploadId')
    vector_store = getattr(app, 'vector_store', None)

    if not question or not user_id or not upload_id:
        return "Missing question, userId, or uploadId in the request body.", 400

    if vector_store is None:
        return "No documents uploaded for context.", 400
    
    fit ={'userId': user_id}
    fit2 = {'uploadId': upload_id}

    try:
        retriever = vector_store.as_retriever(search_kwargs={"filter": fit, "filter": fit2})

        # Ensure the prompt template is correctly formatted
        prompt_template = ChatPromptTemplate.from_template(
            "Context: {context}\nAnswer the following question; if you don't know the answer, say so:\nQuestion: {input}"
        )

        # Initialize the LLM model (OpenAI in this case)
        from langchain_openai import ChatOpenAI
        openai_api_key = os.getenv('OPENAI_API_KEY')
        llm = ChatOpenAI(model="gpt-4-1106-preview", api_key=openai_api_key, temperature=0, max_tokens=4000)

        # Create the document chain and retrieval chain correctly
        document_chain = create_stuff_documents_chain(llm,prompt_template)
        retrieval_chain = create_retrieval_chain(retriever,document_chain)

        # Invoke the chain with the question and context
        response = retrieval_chain.invoke({'input': question})
        
        print (f"Response : {response}")

        return jsonify(f"Response : {response}"), 200

    except Exception as e:
        print(f"Error processing question: {e}")
        return f"Error processing question: {e}", 500
    
  #---------------------Using Custom Agents and Tools for nlp headers attributes -----------------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

def retrieve_header_attributes_answer(question: str, user_id: str, session_id: str) -> str:
    vector_store = getattr(app, 'vector_store', None)
    """Retrieve an answer from the ChromaDB based on the question."""
    print("Entered into retrieve_answer function")
    print(f"Question: {question}")
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    
    fit ={'userId': user_id}
    fit2 = {'uploadId': session_id}
    
    retriever1 = vector_store.as_retriever(search_kwargs={"filter": fit, "filter": fit2})
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
    openai_api_key = os.getenv('OPENAI_API_KEY')
    llm1 = ChatOpenAI(model="gpt-4-1106-preview", 
                     api_key=openai_api_key, 
                     temperature = 0, 
                     max_tokens=4000)
    question_answer_chain_header = create_stuff_documents_chain(llm1, prompt1)
    rag_chain_header = create_retrieval_chain(retriever1, question_answer_chain_header)
    response = rag_chain_header.invoke({"input": question})
    print(f"Response: {response['answer']}")
    # context = response.get("context", "")
    contextloi = response['answer']
    # for document in context:
    #     metadata = document.metadata
    #     # print(document.metadata)
    #     if metadata['user_id'] == user_id and metadata['session_id'] == session_id:
    #         # answer = document.page_content
    #         answer = response.get("answer", "")
    #         break
    #     else:
    #         answer = "No answer found, due to incorrect user_id or session_id"
    # # return response["answer"]
    return contextloi

class CustomAgentHeaderAttributes:
    def __init__(self, tools):
        self.tools = tools

    def invoke(self, input_data):
        print("Entered into Custom Agent invoke function")
        question = input_data.get("input", "")
        user_id = input_data.get("user_id", "")
        session_id = input_data.get("session_id", "")
        print(f"User ID: {user_id}")
        print(f"Session ID: {session_id}")
        print(f"Question: {question}")
        for tool in self.tools:
            print("Type of tool: ", type(tool))
            response = tool(str(question), str(user_id), str(session_id))
            if response:
                return {"answer": response}
        return {"answer": "No answer found"}

tools = [retrieve_header_attributes_answer]
agentHeader = CustomAgentHeaderAttributes(tools)

@app.route("/header_attributes_response", methods=["POST"])
def retrieval_agent_headers_attributes():
    print("Entered into Retrieval Agent hit API is called")
    question_map = request.json.get('question')
    user_id = request.json.get('userId')
    session_id = request.json.get('uploadId')
    print(user_id)
    print(session_id)
    print(f"Questions : {question_map}")
    print("Invoking agent...")
    responses = {}
    for key, value in question_map.items():
        response = agentHeader.invoke({"input": value, "user_id": user_id, "session_id": session_id})
        answer = response.get("answer", "")
        print(f"Answer: {answer}")
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
def retrieve_section_answers(question: str, user_id: str, session_id:str) -> str:
    vector_store = getattr(app, 'vector_store', None)
    """Retrieve an answer from the ChromaDB based on the question."""
    print("Entered into retrieve_answer_res function")
    print(f"Question: {question}")
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    fit ={'userId': user_id}
    fit2 = {'uploadId': session_id}
    
    retriever2 = vector_store.as_retriever(search_kwargs={"filter": fit, "filter": fit2})
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
    openai_api_key = os.getenv('OPENAI_API_KEY')
    llm2 = ChatOpenAI(model="gpt-4-1106-preview", 
                      api_key=openai_api_key, 
                      temperature = 0, 
                      max_tokens=4000)
    question_answer_chain = create_stuff_documents_chain(llm2, prompt2)
    rag_chain = create_retrieval_chain(retriever2, question_answer_chain)
    response = rag_chain.invoke({"input": question})
    # print(f"Response: {response}")
    # context = response.get("context", "")
    contextioe = response['answer']
    # for document in context:
    #     metadata = document.metadata
    #     # print(document.metadata)
    #     if metadata['user_id'] == user_id and metadata['session_id'] == session_id:
    #         # answer = document.page_content
    #         answer = response.get("answer", "")
    #         break
    #     else:
    #         answer = "No answer found, due to incorrect user_id or session_id"
    return contextioe

class RetrievalAgentSectionAnswers:
    def __init__(self, tools):
        self.tools = tools

    def invoke(self, input_data):
        print("Entered into Custom Agent invoke function")
        question = input_data.get("input", "")
        user_id = input_data.get("user_id", "")
        session_id = input_data.get("session_id", "")
        print(f"Question: {question}")
        print(f"User ID: {user_id}")
        print(f"Session ID: {session_id}")
        for tool in self.tools:
            print("Type of tool: ", type(tool))
            response = tool(str(question), str(user_id), str(session_id))
            if response:
                return {"answer": response}
        return {"answer": "No answer found"}

retrieval_tools = [retrieve_section_answers]
retrieval_agent = RetrievalAgentSectionAnswers(retrieval_tools)

@app.route("/retrieve_sections_answers", methods=["POST"])
def retrieval_agent_section_answers():
    print("Entered into Retrieval Agent hit API is called")
    question = request.json.get('question')
    user_id = request.json.get('userId')
    session_id = request.json.get('uploadId')
    print(f"Questions : {question}")
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    print("Invoking agent...")
    responses = {}
    for value in question:
        response = retrieval_agent.invoke({"input": value, "user_id": user_id, "session_id": session_id})
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
    nlp_output = process_sections_result(result_sections_dict)
    return json.dumps(result, indent=2), 200
    # return nlp_output, 200

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
            continue  # Skip "Problem List" because it's already added

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
    return final_result_json

#------------------------- Using custom tool for schema arrangement for the llm response ----------------------------------------------
from langchain_openai import ChatOpenAI
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
    openai_api_key = os.getenv('OPENAI_API_KEY')
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

@app.route("/get_nlp_schema", methods=["POST"])
def format_medical_problems():
    print("1. Entered into format_medical_problems hit API is called")
    import json
    data = request.json  # Get the entire JSON data
    print(f"Data: {data}")

    # Extract the Problem List
    problem_list = data.get("Problem List", [])
    print(f"Problem List: {problem_list}")

    # Initialize the output data with the Problem List
    output_data = {
        "problems": problem_list
    }

    # Iterate over the rest of the data to add the medication and instructions
    for key, value in data.items():
        if key != "Problem List":
            output_data["problems"].append(f"{key} {value}")

    # List of problems to be processed
    problems = output_data["problems"]
    print(f"Problems: {problems}")

    responses = []
    for problem in problems:
        response = agent_schema.invoke({"input": problem})
        print(f"Response: {response}")
        responses.append(response.get("answer", ""))

    result = []
    for response in responses:
        content = response.content
        cleaned_response = content.replace("```json", "").replace("```", "").strip()
        print(cleaned_response)
        result.append(json.loads(cleaned_response))

    print("End of format_medical_problems hit API is called")
    print("NLP response schema created successfully!!")
    return jsonify(result), 200

 
# @app.route("/get_nlp_schema", methods=["POST"])
# def format_medical_problems():
#     print(" 1.Entered into format_medical_problems hit API is called")
#     import json
#     list  = request.json.get("Problem List", [])
#     print(f"List: {list}")
#     # problems = final_result.get("problems", [])
#     output_data = {
#     "problems": list["Problem List"]
#         }

#     for medication, instruction in list.items():
#         if medication != "Problem List":
#             output_data["problems"].append(f"{medication} {instruction}")
#     final_list =json.dumps(output_data, indent=2)
#     print(final_list)
#     problems=final_list
#     print(f"Problems: {problems}")
#     responses = []
#     # for problem in problems:
#     #     response = agent_schema.invoke({"input": problem})
#     #     # print(f"Response: {response}")
#     #     responses.append(response.get("answer", ""))
#     # result = []
#     # for response in responses:
#     #     content = response.content
#     #     cleaned_response = content.replace("```json", "").replace("```", "").strip()
#     #     # print(cleaned_response)
#     #     result.append(json.loads(cleaned_response))
#     print("End of format_medical_problems hit API is called")
#     print("NLP response schema created successfully!!")
#     return jsonify("result"), 200



if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(port=2000)
