
#------------------------------- Text Loader -----------------------------------
def text_document_loader(docs, u_id, s_id, dc_id):
    print("1. Entered into textDocumentLoader called function for extracting text from text files")
    # from langchain_community.document_loaders import DirectoryLoader 
    from langchain_community.document_loaders import TextLoader
    import chardet
    import os
    import shutil
    upload_folder = 'uploaded_files'
    if os.path.exists(upload_folder):
        shutil.rmtree(upload_folder)
    os.makedirs(upload_folder)
    for doc in docs:
        file_path = os.path.join(upload_folder, doc.filename)
        with open(file_path, "wb") as f:
            f.write(doc.read())
    with open(file_path, 'rb') as f:
        file_content = f.read()
    encoding = chardet.detect(file_content)['encoding']
    loader = TextLoader(file_path, encoding=encoding) # for loading the text files
        # dir_path = path
        # text_loader_kwargs={'autodetect_encoding': True} # for autodetecting the encoding of the text files
        # loader = DirectoryLoader(
        #                             dir_path,
        #                             glob = "**/*.txt", 
        #                             loader_cls = TextLoader,
        #                             show_progress=True, 
        #                             silent_errors=True,
        #                             loader_kwargs=text_loader_kwargs
        #                         )
    docs = loader.load() 
    for doc in docs: # loop for adding unique_id to each document
        doc_id = dc_id
        user_id = u_id
        session_id = s_id
        doc.metadata['user_id'] = user_id
        doc.metadata['doc_id'] = doc_id
        doc.metadata['session_id'] = session_id
    print(f"Number of documents loaded: {len(docs)}")
    for doc in docs: # loop for printing the metadata of each document
        print(doc.metadata)
    from text_splitters import text_splitter_txtdocs
    chromadb, embedding_model, docs = text_splitter_txtdocs(docs) #Function call for text splitter 
    return chromadb, embedding_model, docs

#------------------------------- PDF Loader -----------------------------------
# from langchain_community.document_loaders import PyPDFLoader
# from uuid import uuid4
# dir_path = r"D:\Edvenswa\LLM POC July\llm-poc-infer\data\colonoscopy report.pdf" # path for the directory containing the documents
# loader = PyPDFLoader(dir_path)
# pages = loader.load_and_split()
# print(len(pages))
# for page in pages: # loop for adding unique_id to each page
#     unique_id = str(uuid4())
#     page.metadata['unique_id'] = unique_id
# print(len(pages))
# for page in pages: # loop for printing the metadata of each page
#     print(page.metadata)

# #Function call for text splitter 
# from scripts.text_splitters import textSplitterPdf
# textSplitterPdf(pages)






