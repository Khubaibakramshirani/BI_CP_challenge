# Use the latest Python 3.12 image
FROM python:3.12

# Set working directory to root (/app)
WORKDIR /app

# Copy requirements file from root
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional system dependencies
RUN apt-get update && apt-get install -y tesseract-ocr libmagic1 poppler-utils

# Copy the backend folder into /app
COPY backend /app/backend

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI using backend.main:app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
