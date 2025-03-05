# Ensure the path exists
import os
import logging
import pickle
from pathlib import Path

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logging.getLogger("httpx").setLevel(logging.WARNING)

# if not os.path.exists(drive_path):
#     raise FileNotFoundError(f"Data directory not found at {drive_path}")

# # Use the pickle name directly instead of constructing it
# pickle_path = os.path.join(drive_path, pickle_name)

# if not os.path.exists(pickle_path):
#     raise FileNotFoundError(f"Pickle file not found at {pickle_path}")

# logger.info(f"Loading data from {pickle_path}"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)         
            
def get_retriever(document_type: str):
    base_path = Path(__file__).parent / "data"
    
    logger.info(f"get_retriever to manage different retrievers. Base path: {base_path}")
    logger.info(f"File Name =  {document_type}")
    
    # Convert document_type to proper folder name
    if document_type == "presentation":
        folder_name = "Presentation"
        pickle_name = "Presentation.pkl"
    elif document_type == "proxy_statement":
        folder_name = "Proxy_Statement"
        pickle_name = "Proxy_Statement.pkl"
    else:
        raise ValueError(f"Unknown document type: {document_type}")
    
    drive_path = base_path / folder_name
    
    # Log the complete path for debugging
    logger.info(f"Attempting to load data from: {drive_path}")
    logger.info(f"Expected pickle file path: {drive_path / pickle_name}")
    
    # Pass both the drive path and pickle name to load_complete_data_from_drive
    #retriever, saved_data = load_complete_data_from_drive(str(drive_path), pickle_name)
    #return retriever
    
    
get_retriever("presentation")
#print(retriever)
    