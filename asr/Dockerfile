# Use a lightweight Python image
FROM python:3.11.1-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only essential files
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the main ASR API
COPY asr_api.py /app/

# Ensure storage for downloaded models (avoids redownloading on every build)
ENV TRANSFORMERS_CACHE=/app/models
RUN mkdir -p /app/models


# Expose FastAPI port
EXPOSE 8001

# Run FastAPI application
CMD ["uvicorn", "asr_api:app", "--host", "0.0.0.0", "--port", "8001"]
