def text_splitter_txtdocs(docs):
    print("2. entered into text splitter Text called function")
    print(f"Number of documents loaded: {len(docs)}")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= 400, # number of characters per chunk
        chunk_overlap=100, # number of characters to overlap between chunks
        # length_function=len, # function to get the length of the text
        # is_separator_regex = False # if True, the separator is a regex pattern
    )
    all_text_chunks = []     # Initialize an empty list to store all chunks
    for doc in docs: # loop for splitting the documents using split_documents method
        doc_id = doc.metadata['doc_id']
        sess_id = doc.metadata['session_id']
        user_id = doc.metadata['user_id']
        print(doc_id)
        print(sess_id)
        print(user_id)
        # print(doc)
        text_chunks = text_splitter.split_documents([doc])
        print(f"Number of Text Chunks created : {len(text_chunks)}")
        all_text_chunks.extend(text_chunks)   # Extend the all_texts list with the new chunks
    from chromadb_practise import using_chromadb
    chromadb, embedding_model, docs = using_chromadb(all_text_chunks, docs) #Function call for using chromaDB
    return chromadb, embedding_model, docs
        
    # for doc in docs: # loop for splitting the documents using create_documents method
    #     doc_id = doc.metadata['unique_id']
    #     print(doc_id)
    #     # print(doc)
    #     texts = text_splitter.create_documents([doc.page_content])
    #     print(len(texts))

#-------------------------text splitter for pdf pages ------------------------------
# def text_splitter_pdfdocs(pages):
#     print("Entered into PDF Text Splitter called function")
#     print(len(pages))
#     from langchain_text_splitters import RecursiveCharacterTextSplitter
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=200, # number of characters per chunk
#         chunk_overlap=50, # number of characters to overlap between chunks
#         length_function=len, # function to get the length of the text
#         is_separator_regex = False # if True, the separator is a regex pattern
#     )
#     for page in pages: # loop for splitting the pages using split_documents method
#         page_id = page.metadata['unique_id']
#         print(page_id)
#         # print(page)
#         texts = text_splitter.split_documents([page])
#         print(len(texts))
#     for page in pages: # loop for splitting the pages using create_documents method
#         page_id = page.metadata['unique_id']
#         print(page_id)
#         # print(page)
#         texts = text_splitter.create_documents([page.page_content])
#         print(len(texts))

       