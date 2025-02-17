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
#from unstructured.partition.pdf import partition_pdf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# print(pytesseract.get_tesseract_version())  # Should print Tesseract version

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


# function to manage different retrievers
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

class QueryRequest(BaseModel):
    question: str
    documentType: str


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

# Optional: Add a health check endpoint
@router_fast_api.get("/health")
async def health_check():
    return {"status": "healthy"}