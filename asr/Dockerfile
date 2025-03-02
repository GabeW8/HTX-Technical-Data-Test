# Use official Python image
FROM python:3.11.1-slim

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create an uploads directory (optional)
RUN mkdir -p /app/audio_samples

# Expose port 8001
EXPOSE 8001

# Run FastAPI with Uvicorn
CMD ["uvicorn", "asr_api:app", "--host", "0.0.0.0", "--port", "8001"]
