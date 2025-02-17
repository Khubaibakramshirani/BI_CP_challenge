# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# import os
# import time
# import uuid
# import logging
# from dotenv import load_dotenv
# from typing import Dict, List, Optional
# from base64 import b64decode

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain.storage import InMemoryStore
# from langchain.schema.document import Document
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.messages import HumanMessage
# from unstructured.partition.pdf import partition_pdf

# import pytesseract
# import requests

# response = requests.get("https://api.smith.langchain.com/health")
# print("https://api.smith.langchain.com/health", response.status_code, response.text)


from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import time
import uuid
import logging
import tenacity
import pytesseract
import requests
import pickle

from dotenv import load_dotenv
from typing import Dict, List, Optional
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from unstructured.partition.pdf import partition_pdf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print(pytesseract.get_tesseract_version())  # Should print Tesseract version

# Initialize the FastAPI APIRouter
router_fast_api = APIRouter()

# Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
load_dotenv(dotenv_path=dotenv_path)

# Helper functions
def parse_docs(docs):
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            if hasattr(text_element, 'text') and text_element.text:
                context_text += text_element.text

    # Use document type specific prompt
    if kwargs.get("documentType") == "proxy_statement":
        prompt_template = f"""
        For context: Proxy statements outline various plans for the company to shareholders, 
        from the election of directors, to pay for executives, approval or amendment of equity plans, 
        and even shareholder-sponsored proposals should any be received. Be specific about data, graphs, 
        charts, visualization.

        Using only the following context, which includes text, tables, and possibly images, answer the question.
        Context: {context_text}
        Question: {user_question}
        """
    else:  # presentation
        prompt_template = f"""
        For context: It's about presentation for Analyst and Investor meeting. It contains visualization, graphs, text charts. 
        Be specific about data, graphs, charts, visualization.

        Using only the following context, which includes text, tables, and possibly images, answer the question.
        Context: {context_text}
        Question: {user_question}
        """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            if image:  # Only add non-None images
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                )

    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

def load_complete_data_from_drive(drive_path, document_type):
    """
    Load data from drive with document-type specific pickle file and validate data
    """
    from langchain.schema.document import Document
    
    # Ensure the path exists
    if not os.path.exists(drive_path):
        raise FileNotFoundError(f"Data directory not found at {drive_path}")
    
    # Set pickle filename based on document type
    pickle_filename = f"{document_type}.pkl"
    pickle_path = os.path.join(drive_path, pickle_filename)
    
    print(f"Loading data from {pickle_path}")
    
    # Load the vectorstore
    vectorstore = Chroma(
        persist_directory=drive_path,
        embedding_function=OpenAIEmbeddings(),
        collection_name="multi_modal_rag"
    )
    print("**\n\nLoaded the vectorstore**\n\n")
    # Load the stored data
    with open(pickle_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    # Create the storage layer and retriever
   # Create the storage layer and retriever
    store = InMemoryStore()

    # Initialize the retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id",
    )

    # Add text summaries to vectorstore
    summary_texts = [
        Document(page_content=summary, metadata={"doc_id": saved_data['doc_ids'][i]})
        for i, summary in enumerate(saved_data['text_summaries'])
    ]
    retriever.vectorstore.add_documents(summary_texts)

    # Add table summaries to vectorstore
    if saved_data['table_summaries']:

        summary_tables = [
            Document(page_content=summary, metadata={"doc_id": saved_data['table_ids'][i]})
            for i, summary in enumerate(saved_data['table_summaries'])
        ]
        retriever.vectorstore.add_documents(summary_tables)

    # Add image summaries to vectorstore
    summary_img = [
        Document(page_content=summary, metadata={"doc_id": saved_data['img_ids'][i]})
        for i, summary in enumerate(saved_data['image_summaries'])
    ]
    retriever.vectorstore.add_documents(summary_img)

    # Restore the connections between summaries and original content
    retriever.docstore.mset(list(zip(saved_data['doc_ids'], saved_data['texts'])))
    retriever.docstore.mset(list(zip(saved_data['table_ids'], saved_data['tables'])))
    retriever.docstore.mset(list(zip(saved_data['img_ids'], saved_data['images'])))

    print(f"Loaded complete data from {drive_path}")

    return retriever, saved_data


# Add this function to manage different retrievers
def get_retriever(document_type: str):
    base_path = Path(__file__).parent / "data"
    print("get_retriever to manage different retrievers")
    # Convert document_type to proper folder name and filename
    if document_type == "presentation":
        folder_name = "Presentation"
    elif document_type == "proxy_statement":
        folder_name = "Proxy_Statement"
    else:
        raise ValueError(f"Unknown document type: {document_type}")
    
    drive_path = base_path / folder_name
    
    # Pass both the drive path and document type to load_complete_data_from_drive
    retriever, saved_data = load_complete_data_from_drive(str(drive_path), folder_name)
    return retriever#, saved_data

# Initialize chain
#retriever = initialize_retriever()

# # Get the appropriate retriever based on document type
# retriever, saved_data = get_retriever(request.documentType)

# chain_with_sources = {
#     "context": retriever | RunnableLambda(parse_docs),
#     "question": RunnablePassthrough(),
# } | RunnablePassthrough().assign(
#     response=(
#         RunnableLambda(build_prompt)
#         | ChatOpenAI(model="gpt-4o-mini")
#         | StrOutputParser()
#     )
# )

# Update your QueryRequest model in rag.py
class QueryRequest(BaseModel):
    question: str
    documentType: str



# Update your route handler
# Update the route handler to pass document type to the chain
@router_fast_api.post("/query")
async def process_query(request: QueryRequest):
    try:
        if not request.question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Processing query for document type: {request.documentType}")
        logger.info("**process_query**")
        # Get the appropriate retriever based on document type
        #retriever, saved_data = get_retriever(request.documentType)
        retriever = get_retriever(request.documentType)
        print("\n**Retriever created**\n")
        # Create a new chain with the selected retriever
        chain_with_sources = {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
            "documentType": RunnableLambda(lambda _: request.documentType),
        } | RunnablePassthrough().assign(
            response=(
                RunnableLambda(build_prompt)
                | ChatOpenAI(model="gpt-4o-mini")
                | StrOutputParser()
            )
        )
        print("\n**Chain with Sources created**\n")
        response = chain_with_sources.invoke(request.question)
        
        if not response or 'response' not in response:
            raise HTTPException(status_code=500, detail="Invalid response from chain")
            
        return {"response": response['response']}
    except Exception as e:
        logger.error(f"Error processing query**: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query??: {str(e)}"
        )
        
    
# # Define PDF paths
# print("path")
# PRESENTATION_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/presentation.pdf"))
# PROXY_STATEMENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/proxy_statement.pdf"))
# print("PRESENTATION_PATH", PRESENTATION_PATH)

# Path resolution

# current_file = __file__
# current_dir = os.path.dirname(current_file)
# data_dir = os.path.join(current_dir, "../data")
# PRESENTATION_PATH = os.path.abspath(os.path.join(current_dir, "../data/presentation.pdf"))

# # Add debug logging
# print(f"Current file: {current_file}")
# print(f"Current directory: {current_dir}")
# print(f"Data directory: {data_dir}")
# print(f"Presentation path: {PRESENTATION_PATH}")

# # Check if the file exists
# if not os.path.exists(PRESENTATION_PATH):
#     print(f"Presentation PDF not found at: {PRESENTATION_PATH}")
# else:
#     print(f"Found presentation PDF at: {PRESENTATION_PATH}")

# # Before initializing RAGPipeline
# if not os.path.isfile(PRESENTATION_PATH):
#     raise FileNotFoundError(f"Presentation PDF not found at: {PRESENTATION_PATH}")

# class QueryRequest(BaseModel):
#     question: str

# # Initialize retriever (do this at startup)
# def initialize_retriever():
#     try:
#         # Get the current file's directory (routers folder)
#         current_dir = Path(__file__).parent 
        
#         # Navigate to the data folder (go up one level then into data)
#         data_dir = current_dir / "data"
        
#         # Ensure the path exists
#         if not data_dir.exists():
#             raise FileNotFoundError(f"Data directory not found at {data_dir}")
            
#         # Initialize vectorstore
#         vectorstore = Chroma(
#             collection_name="multi_modal_rag",
#             embedding_function=OpenAIEmbeddings(),
#             persist_directory=str(data_dir / "chroma_db")  # Point to chroma_db folder
#         )
        
#         # Load docstore
#         docstore_path = data_dir / "presentation_retriever_docstore.pkl"
#         if not docstore_path.exists():
#             raise FileNotFoundError(f"Docstore file not found at {docstore_path}")
            
#         with open(docstore_path, "rb") as f:
#             docstore = pickle.load(f)
        
#         print(f"Initialize retriever - loaded from {data_dir}")
        
#         return MultiVectorRetriever(
#             vectorstore=vectorstore,
#             docstore=docstore,
#             id_key="doc_id",
#         )
#     except Exception as e:
#         print(f"Error initializing retriever: {str(e)}")
#         raise

####--------------------------------------------------------------------



# # Constants
# BATCH_SIZE = 5  # Number of items to process in parallel
# MAX_RETRIES = 3
# RETRY_DELAY = 10  # seconds

# class RateLimitedChatOpenAI(ChatOpenAI):
#     @tenacity.retry(
#         stop=tenacity.stop_after_attempt(MAX_RETRIES),
#         wait=tenacity.wait_exponential(multiplier=RETRY_DELAY, min=4, max=60),
#         retry=tenacity.retry_if_exception_type(Exception),
#         before_sleep=lambda retry_state: logger.info(f"Retrying after {retry_state.next_action.sleep} seconds...")
#     )
#     def generate(self, *args, **kwargs):
#         return super().generate(*args, **kwargs)

# class RAGPipeline:
#     def __init__(self, PRESENTATION_PATH):
#         print("\n[DEBUG] Entering __init__")
#         print(f"[DEBUG] Received presentation_path: {PRESENTATION_PATH}")
#         self.presentation_path = PRESENTATION_PATH
#         print("[DEBUG] File exists, continuing initialization")
        
#         print("[DEBUG] Initializing LLM ")
#         self.llm = RateLimitedChatOpenAI(
#             model="gpt-4o-mini",
#             openai_api_key=os.getenv("OPENAI_API_KEY"),
#             request_timeout=60,
#             max_retries=MAX_RETRIES,
#         )
#         print("[DEBUG] Initializing vector stores")
#         self.vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())
#         self.docstore = InMemoryStore()
#         self.retriever = MultiVectorRetriever(
#             vectorstore=self.vectorstore,
#             docstore=self.docstore,
#             id_key="doc_id"
#         )
#         print("\n[DEBUG] Entering _initialize_text_prompt")
#         self.text_prompt = ChatPromptTemplate.from_template("""
#         You are an assistant tasked with summarizing text.
#         Give a concise summary.
#         Respond only with the summary, no additional comment.
#         Text chunk: {element}
#         """)
#         print("\n[DEBUG] Entering _initialize_table_prompt")
#         self.table_prompt = ChatPromptTemplate.from_template("""
#         You are an assistant tasked with summarizing tables.
#         Provide a concise summary of the table.
#         Respond only with the summary, no additional comment.
#         Table content: {element}
#         """)
#         print("\n[DEBUG] Entering _initialize_image_prompt")
#         self.image_prompt = ChatPromptTemplate.from_messages([
#             ("user", [
#                 {"type": "text", "text": "Describe the image in detail. For context, it is part of a research paper explaining the transformer architecture."},
#                 {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
#             ])
#         ])
#         print("\n[DEBUG] Entering _initialize_chunks from partition_pdf")
#         self.chunks = partition_pdf(
#             filename=PRESENTATION_PATH,
#             infer_table_structure=True,
#             strategy="hi_res",
#             extract_image_block_types=["Image"],
#             extract_image_block_to_payload=True,
#             chunking_strategy="by_title",
#             max_characters=10000,
#             combine_text_under_n_chars=2000,
#             new_after_n_chars=6000,
#         )
#         print("\n[DEBUG] chunks from partition_pdf created\nAssigning self.tables, self.texts, self.images")
        
#         self.tables, self.texts, self.images = [], [], []
#         for chunk in self.chunks:
#             if "Table" in str(type(chunk)):
#                 self.tables.append(chunk)
#             elif "CompositeElement" in str(type(chunk)):
#                 self.texts.append(chunk)
#                 for el in chunk.metadata.orig_elements:
#                     if "Image" in str(type(el)):
#                         self.images.append(el.metadata.image_base64)
        
#         self.store_documents()
    
#     def summarize_documents(self):
#         """
#         Summarize text, table, and image documents in parallel using thread pools.
#         Returns tuples of summaries for each document type.
#         """
#         print("\n[DEBUG] Entering summarize_documents(self): - Line ~190")
#         def process_text_batch(batch):
#             print(f"[DEBUG] Processing text batch of size {len(batch)}")
#             chain = self.text_prompt | self.llm | StrOutputParser()
#             return [chain.invoke({"element": item.text}) for item in batch]
        
#         def process_table_batch(batch):
#             print(f"[DEBUG] Processing table batch of size {len(batch)}")
#             chain = self.table_prompt | self.llm | StrOutputParser()
#             return [chain.invoke({"element": item.text}) for item in batch]
        
#         def process_image_batch(batch):
#             print(f"[DEBUG] Processing image batch of size {len(batch)}")
#             chain = self.image_prompt | self.llm | StrOutputParser()
#             return [chain.invoke({"image": img}) for img in batch]
        
#         def batch_processor(items, process_func):
#             if not items:
#                 return []
            
#             summaries = []
#             with ThreadPoolExecutor() as executor:
#                 # Process items in batches
#                 for i in range(0, len(items), BATCH_SIZE):
#                     batch = items[i:i + BATCH_SIZE]
#                     try:
#                         batch_summaries = process_func(batch)
#                         summaries.extend(batch_summaries)
#                     except Exception as e:
#                         logger.error(f"Error processing batch: {e}")
#                         # Add empty summaries for failed batch to maintain alignment
#                         summaries.extend(["" for _ in batch])
                    
#                     # Add a small delay between batches to avoid rate limiting
#                     if i + BATCH_SIZE < len(items):
#                         time.sleep(1)
            
#             return summaries
        
#         # Process each document type in parallel
#         print("[DEBUG] Starting batch processing")
#         text_summaries = batch_processor(self.texts, process_text_batch)
#         table_summaries = batch_processor(self.tables, process_table_batch)
#         image_summaries = batch_processor(self.images, process_image_batch)
#         print("[DEBUG] Completed all batch processing")
#         return text_summaries, table_summaries, image_summaries

#     def store_documents(self):
#         print("\n[DEBUG] Entering store_documents - Line ~220")
#         try:
#             print("[DEBUG] Calling summarize_documents()")
#             text_summaries, table_summaries, image_summaries = self.summarize_documents()
            
#             print("[DEBUG] Storing documents in vector store")
#             if text_summaries and self.texts:
#                 print(f"[DEBUG] Storing {len(text_summaries)} text summaries")
#                 doc_ids = [str(uuid.uuid4()) for _ in self.texts]
#                 summary_texts = [
#                     Document(page_content=summary, metadata={"doc_id": doc_ids[i]}) 
#                     for i, summary in enumerate(text_summaries)
#                 ]
#                 self.retriever.vectorstore.add_documents(summary_texts)
#                 self.retriever.docstore.mset(list(zip(doc_ids, self.texts)))
            
#             if table_summaries and self.tables:
#                 print(f"[DEBUG] Storing {len(table_summaries)} table summaries")
#                 table_ids = [str(uuid.uuid4()) for _ in self.tables]
#                 summary_tables = [
#                     Document(page_content=summary, metadata={"doc_id": table_ids[i]}) for i, summary in enumerate(table_summaries)
#                 ]
#                 self.retriever.vectorstore.add_documents(summary_tables)
#                 self.retriever.docstore.mset(list(zip(table_ids, self.tables)))
            
#             if image_summaries and self.images:
#                 print(f"[DEBUG] Storing {len(image_summaries)} image summaries")
#                 img_ids = [str(uuid.uuid4()) for _ in self.images]
#                 summary_img = [
#                     Document(page_content=summary, metadata={"doc_id": img_ids[i]}) for i, summary in enumerate(image_summaries)
#                 ]
#                 self.retriever.vectorstore.add_documents(summary_img)
#                 self.retriever.docstore.mset(list(zip(img_ids, self.images)))
            
#             print("[DEBUG] Successfully completed document storage")
            
#         except Exception as e:
#             print(f"[DEBUG] Error in store_documents: {str(e)}")
#             print(f"[DEBUG] Error type: {type(e)}")
#             print(f"[DEBUG] Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
#             raise
        
#     def parse_docs(self, docs):
#         images, texts = [], []
#         for doc in docs:
#             try:
#                 b64decode(doc)
#                 images.append(doc)
#             except Exception:
#                 texts.append(doc)
#         return {"images": images, "texts": texts}
    
#     def build_prompt(self, kwargs):
#         docs_by_type = kwargs["context"]
#         user_question = kwargs["question"]
        
#         context_text = "".join(text.text for text in docs_by_type["texts"])
#         prompt_template = f"""
#         Answer the question based only on the following context.
#         Context: {context_text}
#         Question: {user_question}
#         """
#         prompt_content = [{"type": "text", "text": prompt_template}]
#         for image in docs_by_type["images"]:
#             prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
        
#         return ChatPromptTemplate.from_messages([{"role": "user", "content": prompt_content}])
    
#     def query_rag(self, question: str):
#         rag_chain = (
#             {"context": self.retriever | RunnableLambda(self.parse_docs), "question": RunnablePassthrough()}
#             | RunnableLambda(self.build_prompt)
#             | self.llm
#             | StrOutputParser()
#         )
#         return rag_chain.invoke(question)

# # # Pydantic model for the request
# class QueryRequest(BaseModel):
#     question: str
    
# rag_pipeline = RAGPipeline(PRESENTATION_PATH)

# @router_fast_api.post("/query")
# async def process_query(request: QueryRequest):
#     response = rag_pipeline.query_rag(request.question)
#     return {"response": response}

# class RateLimitedChatOpenAI(ChatOpenAI):
#     @tenacity.retry(
#         stop=tenacity.stop_after_attempt(MAX_RETRIES),
#         wait=tenacity.wait_exponential(multiplier=RETRY_DELAY, min=4, max=60),
#         retry=tenacity.retry_if_exception_type(Exception),
#         before_sleep=lambda retry_state: logger.info(f"Retrying after {retry_state.next_action.sleep} seconds...")
#     )
#     def generate(self, *args, **kwargs):
#         return super().generate(*args, **kwargs)

# # Initialize language model with retry capability
# llm = RateLimitedChatOpenAI(
#     model="gpt-4o-mini",  # Using a more stable model
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
#     request_timeout=60,
#     max_retries=MAX_RETRIES,
# )
# print("LLM initialized")

# #def process_pdf(file_path: str) -> tuple:
# #   """Process PDF and return chunks of different types."""
# try:
#     chunks = partition_pdf(
#         filename=PRESENTATION_PATH,
#         infer_table_structure=True,
#         strategy="hi_res",
#         extract_image_block_types=["Image"],
#         extract_image_block_to_payload=True,
#         chunking_strategy="by_title",
#         max_characters=10000,
#         combine_text_under_n_chars=2000,
#         new_after_n_chars=6000,
#     )
    
#     print("Chunking done")
#     tables, texts, images = [], [], []
    
#     for chunk in chunks:
#         if "Table" in str(type(chunk)):
#             tables.append(chunk)
            
#         elif "CompositeElement" in str(type(chunk)):
#             texts.append(chunk)
            
#             # Extract images from composite elements
#             for el in chunk.metadata.orig_elements:
#                 if "Image" in str(type(el)):
#                     images.append(el.metadata.image_base64)
                    
#     print("Chunks added in table")              
# except Exception as e:
#     logger.error(f"PDF processing failed: {str(e)}")
#     raise

# def batch_process(items: List, process_func, batch_size: int = BATCH_SIZE):
#     """Process items in batches to avoid rate limits."""
#     results = []
#     for i in range(0, len(items), batch_size):
#         print(f"{i} batch process")
#         batch = items[i:i + batch_size]
#         try:
#             batch_results = process_func.batch(batch, {"max_concurrency": batch_size})
#             results.extend(batch_results)
#             time.sleep(1)  # Small delay between batches
#         except Exception as e:
#             logger.error(f"Batch processing failed: {str(e)}")
#             raise
#     return results

# # Summarization chains
# prompt_text = """
# You are an assistant tasked with summarizing tables and text.
# Give a concise summary of the table or text.

# Respond only with the summary, no additional comment.
# Do not start your message by saying "Here is a summary".
# Just give the summary as it is.

# Table or text chunk: {element}
# """
# prompt_text_table = ChatPromptTemplate.from_template
# summarize_chain = {"element": lambda x: x} | prompt_text_table | llm | StrOutputParser()

# # Summarize text
# text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

# # Summarize tables
# tables_html = [table.metadata.text_as_html for table in tables]
# table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})

# prompt_image = """Describe the image in detail. For context,
#                   the image is part of a research paper explaining the transformers
#                   architecture. Be specific about graphs, such as bar plots."""
# messages = [
#     (
#         "user",
#         [
#             {"type": "text", "text": prompt_image},
#             {
#                 "type": "image_url",
#                 "image_url": {"url": "data:image/jpeg;base64,{image}"},
#             },
#         ],
#     )
# ]

# prompt_img_chain = ChatPromptTemplate.from_messages(messages)
# chain_img = prompt_img_chain | llm | StrOutputParser()
# image_summaries = chain_img.batch(images)


# # The vectorstore to use to index the child chunks
# vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())

# # The storage layer for the parent documents
# store = InMemoryStore()
# id_key = "doc_id"

# # The retriever (empty to start)
# retriever = MultiVectorRetriever(
#     vectorstore=vectorstore,
#     docstore=store,
#     id_key=id_key,
# )

# # Add texts
# if text_summaries and texts:
#     doc_ids = [str(uuid.uuid4()) for _ in texts]
#     summary_texts = [
#         Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
#     ]
#     retriever.vectorstore.add_documents(summary_texts)
#     retriever.docstore.mset(list(zip(doc_ids, texts)))

# # Add tables
# if table_summaries and tables:
#     table_ids = [str(uuid.uuid4()) for _ in tables]
#     summary_tables = [
#         Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
#     ]
#     retriever.vectorstore.add_documents(summary_tables)
#     retriever.docstore.mset(list(zip(table_ids, tables)))

# # Add image summaries
# if image_summaries and images:
#     img_ids = [str(uuid.uuid4()) for _ in images]
#     summary_img = [
#         Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
#     ]
#     retriever.vectorstore.add_documents(summary_img)
#     retriever.docstore.mset(list(zip(img_ids, images)))


# def parse_docs(docs):
#     """Split base64-encoded images and texts"""
#     b64 = []
#     text = []
#     for doc in docs:
#         try:
#             b64decode(doc)
#             b64.append(doc)
#         except Exception as e:
#             text.append(doc)
#     return {"images": b64, "texts": text}


# def build_prompt(kwargs):

#     docs_by_type = kwargs["context"]
#     user_question = kwargs["question"]

#     context_text = ""
#     if len(docs_by_type["texts"]) > 0:
#         for text_element in docs_by_type["texts"]:
#             context_text += text_element.text

#     # construct prompt with context (including images)
#     prompt_template = f"""
#     Answer the question based only on the following context, which can include text, tables, and the below image.
#     Context: {context_text}
#     Question: {user_question}
#     """

#     prompt_content = [{"type": "text", "text": prompt_template}]

#     if len(docs_by_type["images"]) > 0:
#         for image in docs_by_type["images"]:
#             prompt_content.append(
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/jpeg;base64,{image}"},
#                 }
#             )

#     return ChatPromptTemplate.from_messages(
#         [
#             HumanMessage(content=prompt_content),
#         ]
#     )

# RAG_chain = (
#     {
#         "context": retriever | RunnableLambda(parse_docs),
#         "question": RunnablePassthrough(),
#     }
#     | RunnableLambda(build_prompt)
#     | llm
#     | StrOutputParser()
# )

# rag_chain_with_sources = {
#     "context": retriever | RunnableLambda(parse_docs),
#     "question": RunnablePassthrough(),
# } | RunnablePassthrough().assign(
#     response=(
#         RunnableLambda(build_prompt)
#         | llm
#         | StrOutputParser()
#     )
# )
        

# response = chain_with_sources.invoke(
#     "What is multihead?"
# )

# print("Response:", response['response'])

# print("\n\nContext:")
# for text in response['context']['texts']:
#     print(text.text)
#     print("Page number: ", text.metadata.page_number)
#     print("\n" + "-"*50 + "\n")
# for image in response['context']['images']:
#     display_base64_image(image)


# # Pydantic model for the request
# class QueryRequest(BaseModel):
#     question: str

# # Process query function with the router
# import time

# @router_fast_api.post("/query")
# async def process_query(request: QueryRequest):
#     try:
#         start_time = time.time()

        
        
#         # Retrieve documents related to the question
#         retrieved_docs = retriever.vectorstore.similarity_search(request.question)
#         retrieval_time = time.time()
#         print(f"Document retrieval took: {retrieval_time - start_time} seconds")

#         # Format the retrieved documents for the RAG chain
#         formatted_docs = format_docs(retrieved_docs)

#         # Set up the input dictionary for the chain
#         input_dict = {"context": formatted_docs, "question": request.question}

#         # Create a prompt and pass it to the language model
#         chain = (
#             {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
#             | prompt
#             | llm
#             | StrOutputParser()
#         )

#         # Get the answer from the RAG chain
#         result = chain.invoke(input_dict)
#         end_time = time.time()
#         print(f"Total processing time: {end_time - start_time} seconds")

#         return {"result": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

# class RAGPipeline:
#     _instance = None
    
#     @classmethod
#     def get_instance(cls):
#         if cls._instance is None:
#             cls._instance = cls()
#         return cls._instance
    
#     def __init__(self):
#         self.initialized = False
#         self.vectorstore = Chroma(
#             collection_name="multi_modal_rag",
#             embedding_function=OpenAIEmbeddings()
#         )
#         self.store = InMemoryStore()
#         self.id_key = "doc_id"
#         self.retriever = MultiVectorRetriever(
#             vectorstore=self.vectorstore,
#             docstore=self.store,
#             id_key=self.id_key
#         )
        
#     def initialize(self):
#         """Initialize the RAG pipeline with document processing."""
#         if self.initialized:
#             return
            
#         logger.info("Initializing RAG pipeline...")
        
#         # Process both PDFs
#         #for pdf_path in [PRESENTATION_PATH, PROXY_STATEMENT_PATH]:
#         # if not os.path.exists(PRESENTATION_PATH):
#         #     logger.error(f"PDF file not found: {PRESENTATION_PATH}")
#         #     continue
            
#         # logger.info(f"Processing PDF: {PRESENTATION_PATH}")
#         try:
#             print("Initialize the RAG pipeline with document processing")
#             tables, texts, images = process_pdf(PRESENTATION_PATH)
#             print("process_pdf complete")
#             # Process in batches
#             summarize_chain = create_summarization_chain()
#             image_chain = create_image_chain()
            
#             text_summaries = batch_process(texts, summarize_chain)
#             table_summaries = batch_process(
#                 [table.metadata.text_as_html for table in tables],
#                 summarize_chain
#             )
#             image_summaries = batch_process(images, image_chain)
#             print("summarization complete")
#             # Add to store
#             self._add_to_store(text_summaries, texts)
#             self._add_to_store(table_summaries, tables)
#             self._add_to_store(image_summaries, images)
#             print("Added to store complete")

#             logger.info(f"Successfully processed PDF: {PRESENTATION_PATH}")
#         except Exception as e:
#             logger.error(f"Error processing PDF {PRESENTATION_PATH}: {str(e)}")
#             #continue
    
#         self.initialized = True
#         logger.info("RAG pipeline initialization complete")

    # ... (keep the rest of the RAGPipeline class methods)

# FastAPI endpoint
# class QueryRequest(BaseModel):
#     question: str



# @router_fast_api.post("/query")
# async def process_query(request: QueryRequest):
#     try:
#         start_time = time.time()
#         print("process query")
        
#         # Get or create RAG pipeline instance
#         rag_pipeline = RAGPipeline.get_instance()
#         print("RAG pipeline initialized")
        
#         # Initialize if not already initialized
#         if not rag_pipeline.initialized:
#             # Create data directory if it doesn't exist
#             data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
#             os.makedirs(data_dir, exist_ok=True)
            
#             # Initialize the pipeline
#             rag_pipeline.initialize()
        
#         # Process query
#         response = rag_pipeline.query(request.question)
#         end_time = time.time()
        
#         return {
#             "result": response,
#             "processing_time": end_time - start_time
#         }
#     except Exception as e:
#         logger.error(f"Query processing failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# Optional: Add a health check endpoint
@router_fast_api.get("/health")
async def health_check():
    return {"status": "healthy"}