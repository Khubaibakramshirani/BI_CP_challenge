# main.py
import os
from fastapi import FastAPI
from backend.routers import rag
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://127.0.0.1:3000",
        "https://rag-app-13737131325.europe-west1.run.app",  # Your production URL
        "*"  # This allows any origin (use only for development)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a root route
@app.get("/health")
def read_root():
    return {"message": "Welcome to the RAG Query System v1.1"}

# Include the RAG router
app.include_router(rag.router_fast_api, prefix="/api")

app.mount("/static", StaticFiles(directory="backend/web/static", html=True), name="static")

app.mount("/", StaticFiles(directory="backend/web", html=True), name="staticweb")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use PORT env variable or default to 8000
    
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)