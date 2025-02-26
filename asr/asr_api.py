from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf
import io
import librosa
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Load ASR model
model_name = os.getenv("MODEL_NAME", "facebook/wav2vec2-large-960h") 
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Maximum file size (e.g., 10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # in bytes

@app.get("/ping")
def ping():
    """Health check endpoint"""
    return {"message": "pong"}

@app.post("/asr")
async def transcribe_audio(file: UploadFile = File(...)):
    """Processes an uploaded audio file and returns its transcription and duration"""
    try:
        # Read the entire file
        audio_bytes = await file.read()
        
        # Check file size using the byte length
        if len(audio_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large. Max size is 10MB.")

        # Use soundfile to load the audio file from bytes
        with io.BytesIO(audio_bytes) as audio_buffer:
            audio_data, sample_rate = sf.read(audio_buffer)
        
        # Convert stereo to mono if necessary (averaging channels)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # If audio is not 16kHz, resample it using librosa
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000  # Update sample rate

        # Calculate the duration of the audio file in seconds
        duration = round(len(audio_data) / sample_rate, 2)

        # Convert the audio data to tensor format expected by the model
        input_values = processor(audio_data, return_tensors="pt", sampling_rate=16000).input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        return {
            "transcription": transcription,
            "duration": str(duration)  # Return duration as a string
        }

    except Exception as e:
        # Log the exception as needed (for production, consider structured logging)
        return {"error": f"An error occurred: {str(e)}"}
