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
from flask_cors import CORS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory 



# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
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
            file_name = file.filename
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

            document = Document(page_content=text_content, metadata={'userId': user_id, 'uploadId': upload_id, "source" : file_name })
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

        return jsonify(response['answer']), 200

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
   Given the medical problem: "{problem}", generate a JSON schema that replicates the structure of the following example, 
    but with new, relevant values based on the input. Do not repeat the example values; 
    instead, generate appropriate content while keeping the structure consistent.
    - `name`: Based on the type of problem (e.g.,DiseaseDisorderMention,LabMention,MedicationMention,ProcedureMention,SignSymptomMention,SectionHeader,AnatomicalSiteMention,EntityMention,MedicalDeviceMention,BacteriumMention,GeneMention)
    - `sectionName`: A relevant section name with character offsets, for example, ["Problem List", 0, 92], depending on the input you are receiving.
    - `sectionOid`: A unique OID for the section, can you please generate a unique OID based on the section name according to HL7 CDA which is a part of CCD? if it is not applicable, you can use 'SIMPLE_SEGMENT'.
    - `sectionOffset`: A simulated character offset range for the section.
    - `sentence`: A simulated character offset range for the sentence.
    - `extendedSentence`: A simulated character offset range for the extended sentence.
    - `text`: The text of the problem with simulated character offsets.
    - `attributes`: A set of attributes such as `fasting`, `derivedGeneric`, `polarity`, `relTime`,`family member`, `confidence`,`secondary`, `date`, `status`, etc., with dynamically generated values relevant to the problem.
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
            "fasting": "True or False",  // Indicates if the patient was fasting
            "derivedGeneric": "1 or 0",  // Indicates if the term is generic
            "polarity": "positive or negated",  // Polarity of the mention
            "relTime": "current status, history status, family history status, probably status",  // Time relation of the problem
            "FamilyMember": "family member if applicable",  // Family member associated with the problem
            "confidence": "0 or 1",  // Confidence level of the problem
            "secondary": "True or False",  // Indicates if the problem is secondary
            "date": "MM-DD-YYYY",  // Date associated with the problem
            "status": "stable, unstable, controlled, not controlled, deteriorating, getting worse, improving, resolved, resolving, unresolved, worsening, well-controlled, unchanges, chronic, diminished, new diagnosis",  // Status of the problem
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
    # openai_api_key = os.getenv('OPENAI_API_KEY')
    # llm3 = ChatOpenAI(model="gpt-4o",
    #                   api_key=openai_api_key,
    #                   temperature=0,
    #                   max_tokens=4000)  
    from langchain_groq import ChatGroq
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm3 = ChatGroq(groq_api_key=groq_api_key, model='Llama3-70b-8192', temperature=0.1, max_tokens=8000) # groq model

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

import re

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
        print(f"Raw Response: {response}")

        # Extract the answer content from the AIMessage
        if response and 'answer' in response:
            answer_content = response["answer"].content

            # Use a regular expression to find the JSON block within the content
            json_match = re.search(r'\{.*\}', answer_content, re.DOTALL)

            if json_match:
                json_schema = json_match.group()
                try:
                    # Parse the JSON string
                    parsed_json = json.loads(json_schema)
                    responses.append(parsed_json)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    print(f"Problematic content: {json_schema}")
                    continue
            else:
                print("No JSON block found in the content.")
                continue
        else:
            print(f"Invalid or empty response for problem: {problem}")

    print("End of format_medical_problems hit API is called")
    print("NLP response schema created successfully!!")
    return jsonify(responses), 200

@app.route("/summarize", methods=["POST"])
def get_summarization():
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
        retriever3 = vector_store.as_retriever(search_kwargs={"filter": fit, "filter": fit2})

        # Ensure the prompt template is correctly formatted
        prompt_template = ChatPromptTemplate.from_template(
        """
        You are an advanced AI language model specialized in deep document analysis and comprehensive summarization.
        I have a set of documents that contain important information relevant to the following question: "{input}".
    
        Based on the context retrieved from the documents, provide an in-depth and exhaustive summary that:
        1. **Thoroughly examines** all main ideas, sections, sub-sections, and supporting details relevant to the question, ensuring nothing is overlooked.
        2. **Analyzes and interprets** critical information, highlighting key details and their implications, including any contradictions or inconsistencies within the documents.
        3. **Identifies and discusses** patterns, relationships, correlations, and any underlying themes, and explains their relevance to the question.
        4. **Explores nuances** and subtle points that may be crucial for a comprehensive understanding, including potential biases, assumptions, or gaps in the information.
        5. **Evaluates the significance** of each piece of information in relation to the overall context, drawing connections between different parts of the documents.
        6. **Includes contextual background** or references where necessary to ensure the summary is fully informed and well-rounded.
        7. **Presents the summary in a detailed, logical, and organized manner**, ensuring clarity while addressing every aspect of the question comprehensively.

        Context from the documents:
        {context}

        Provide the detailed and exhaustive summary below:
        """
        )
        # Initialize the LLM model (OpenAI in this case)
        from langchain_groq import ChatGroq
        groq_api_key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(groq_api_key=groq_api_key, model='Llama3-70b-8192', temperature=0.1, max_tokens=8000) # groq model

        # Create the document chain and retrieval chain correctly
        document_chain = create_stuff_documents_chain(llm,prompt_template)
        retrieval_chain = create_retrieval_chain(retriever3,document_chain)

        # Invoke the chain with the question and context
        response = retrieval_chain.invoke({'input': question})
        
        print (f"Response : {response}")

        return jsonify(response['answer']), 200

    except Exception as e:
        print(f"Error processing question: {e}")
        return f"Error processing question: {e}", 500
    
  #---------------------Using Custom Agents and Tools for nlp headers attributes -----------------------------------------------------------------
@app.route('/chronological_order', methods=['POST'])
def arrange_chronologically():
    data = request.get_json()
    userid = data.get('userId')
    sessionid = data.get('uploadId')
    vector_store = getattr(app, 'vector_store', None)
    section_names = [
                    "Problem List","Medical History","Medications","Social History","Surgical History",
                    "Family History","Vital Signs","Lab Results","Imaging Results","Pathology Reports"
                    ]
    res_ans_dict = {} # list to store the response['answers'] from the model
    for section_name in section_names:
        query = f"""Given a collection of electronic health records (EHR) stored as documents, I am working on 
                    organizing the entries within the EHR data for the section "{section_name}" chronologically,
                    to facilitate easy access and understanding. The goal is to list the most recent entries first.
                    For this section, please provide a structured summary that includes at least 10 entries, 
                    if available. Each entry must be accompanied by a date to indicate when it was recorded or 
                    updated. This organization is crucial for creating a clear, chronological narrative of the patient's 
                    health history based on the available data.Additionally, it is important to identify and include 
                    the source document's name (either a PDF or a text file) from which each entry is derived. 
                    This will aid in tracing the information back to its original context if needed.Could you 
                    assist by processing the uploaded documents, extracting the relevant information for 
                    "{section_name}", and organizing it as requested? Please ensure to maintain accuracy and 
                    clarity in the chronological arrangement and presentation of the data."""
        # global conversation_chain
        # if conversation_chain is None:
        #     return jsonify({'error': 'Conversation chain not initialized'}), 500
        # response = conversation_chain({'question': query, 'userId': userid, 'uploadId': sessionid})  
        # from langchain_openai import ChatOpenAI
        # openai_api_key = os.getenv('OPENAI_API_KEY')
        # llm = ChatOpenAI(model="gpt-4-1106-preview",api_key=openai_api_key,temperature = 0)
        from langchain_groq import ChatGroq
        groq_api_key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(groq_api_key=groq_api_key, model='Llama3-70b-8192', temperature=0.1) # groq model

        retriever = None
        fit ={'userId': userid}
        fit2 = {'uploadId': sessionid}
        # use_new_groq_api_key() # Function to use a new GROQ API key
        memory = ConversationBufferMemory(memory_key='chat_history',output_key='answer', return_messages=True)
        retriever=vector_store.as_retriever(search_kwargs={"filter": fit, "filter": fit2})
        conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = retriever,
        return_source_documents=True,
        memory=memory
        )
        response = conversation_chain({'question': query})
        # filtered_response = {'answer': '', 'source_documents': []}  
        # Assuming you want to store the response in a dictionary
        res_ans_dict[section_name] = response

    # Convert the response dictionary to a JSON serializable format
    serializable_response = {section: {
                                'answer': res_ans_dict[section].get('answer'),
                                'source_documents': [
                                    {'source': doc.metadata.get('source'), 'content': doc.page_content}
                                    for doc in res_ans_dict[section].get('source_documents', [])
                                ]
                            }
                            for section in res_ans_dict}

    return jsonify(serializable_response), 200


if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(port=2000)
