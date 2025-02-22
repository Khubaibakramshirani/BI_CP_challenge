﻿# PDF-based RAG (Retrieval-Augmented Generation) Project

This repository contains a RAG-based application for processing and analyzing PDF documents, built with FastAPI backend and React frontend.

## Project Overview

The project implements a Retrieval-Augmented Generation (RAG) system for PDF documents, utilizing modern technology stack:
- Frontend: React with TypeScript
- Backend: FastAPI with LangChain
- Document Processing: Tesseract OCR, Poppler

## Project Structure

```
├── backend
│   ├── __pycache__
│   ├── routers
│   │   └── __pycache__
│   ├── data
│   │   ├── Presentation
│   │   ├── Proxy_Statement
│   │   ├── faiss_document_store.db
│   │   ├── presentation.pdf
│   │   └── proxy_statement.pdf
│   ├── rag.py
│   ├── web
│   ├── .env
│   └── main.py
├── frontend
│   ├── node_modules
│   ├── public
│   ├── src
│   │   ├── components
│   │   ├── services
│   │   ├── App.css
│   │   ├── App.tsx
│   │   ├── constants.ts
│   │   ├── index.css
│   │   ├── index.tsx
│   │   ├── logo.svg
│   │   ├── react-app-env.d.ts
│   │   ├── reportWebVitals.ts
│   │   └── setupTests.ts
│   ├── package-lock.json
│   ├── package.json
│   ├── README.md
│   └── tsconfig.json
├── .gitignore
├── build.sh
└── out.log
```

## Prerequisites

Before setting up the project, ensure you have the following installed:
- Docker and Docker Compose
- Node.js and npm
- Python 3.8+
- Tesseract OCR v5.5.0.20241111
- Poppler v24.08.0

### Required System Dependencies

1. **Tesseract OCR**
   - Download from: [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add installation path to system's PATH variable
   - Verify installation: `tesseract --version`

2. **Poppler**
   - Download the latest version
   - Add installation path to system's PATH variable
   - Verify installation: `where pdfinfo`

## Installation

### Running the Project with Docker

The project is containerized using Docker. Follow these steps to build and run it using `docker-compose`:

1. **Ensure Docker is running** on your system.
2. **Navigate to the project directory**:
   ```bash
   cd /path/to/your/project
   ```
3. **Build and run the containers**:
   ```bash
   docker-compose up --build
   ```
   This will:
   - Build the backend and frontend images
   - Install necessary dependencies
   - Start the application

4. **Access the application**:
   - Backend API: `http://localhost:8000/docs`
   - Frontend: `http://localhost:3000`

5. **Stopping the application**:
   ```bash
   docker-compose down
   ```

### Running Without Docker (Local Setup)

#### Frontend Setup

```bash
# Install dependencies
npm install

# Start the frontend server
npm start
```

#### Backend Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix/macOS
   .\venv\Scripts\activate   # Windows
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the backend:
   ```bash
   uvicorn main:app --reload
   ```

## Key Dependencies

### Backend Libraries
- FastAPI: Web framework
- LangChain: RAG implementation
- Unstructured: Document processing
- ChromaDB: Vector store
- Tesseract: OCR processing

### Frontend Libraries
- React: UI framework
- TypeScript: Type safety
- Axios: HTTP client

## Configuration

1. Set up environment variables in `.env` file. It should contain:
   ```
   OPENAI_API_KEY=<your-key>
   LANGCHAIN_API_KEY=<your-key>
   LANGCHAIN_TRACING_V2=false
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_PROJECT=langchain-multimodal
   ```
2. Ensure Tesseract and Poppler paths are correctly set in system PATH
3. Configure any additional model settings in the backend

## Usage

1. Navigate to `http://localhost:3000` in your browser for local deployment.
2. Ask a question using the input box and submit.
3. View the response displayed in a chat-like interface.

## Contributing

Feel free to open issues and submit pull requests for any changes.

