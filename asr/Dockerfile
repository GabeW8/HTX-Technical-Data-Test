# Dockerfile for asr-api

# 1. Use a lightweight Python base image
FROM python:3.11.1-slim

# 2. Create a working directory
WORKDIR /app

# 3. Copy only essential files first (helps leverage Docker’s caching)
COPY requirements.txt /app/

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code
COPY asr_api.py /app/

# 6. Cache for Hugging Face transformers
ENV TRANSFORMERS_CACHE=/app/models
RUN mkdir -p /app/models

# 7. Expose the port used by FastAPI
EXPOSE 8001

# 8. Run the API
CMD ["uvicorn", "asr_api:app", "--host", "0.0.0.0", "--port", "8001"]
