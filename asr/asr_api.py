import os
import io
import torch
import soundfile as sf
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from dotenv import load_dotenv

load_dotenv()  # load environment variables

app = FastAPI()

# load the ASR model from env variable, defaulting if not provided
model_name = os.getenv("MODEL_NAME", "facebook/wav2vec2-large-960h")
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# maximum allowed file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

@app.get("/ping")
def ping():
    """health check endpoint"""
    return {"message": "pong"}

@app.post("/asr")
async def transcribe_audio(file: UploadFile = File(...)):
    """process an uploaded audio file and return its transcription and duration"""
    try:
        # read entire file and check its size
        audio_bytes = await file.read()
        if len(audio_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large. Max size is 10MB.")

        # load audio data using soundfile from an in-memory bytes stream
        with io.BytesIO(audio_bytes) as audio_buffer:
            audio_data, sample_rate = sf.read(audio_buffer)

        # if stereo, convert to mono by averaging channels
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # resample audio to 16kHz if needed
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        # calculate duration in seconds
        duration = round(len(audio_data) / sample_rate, 2)

        # prepare input and run inference
        input_values = processor(audio_data, return_tensors="pt", sampling_rate=16000).input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        return {
            "transcription": transcription,
            "duration": str(duration)
        }
    except Exception as e:
        # return a generic error message; in production consider structured logging
        return {"error": f"An error occurred: {str(e)}"}

# optional testing code; executed only if this module is run directly
if __name__ == "__main__":
    # for quick testing of the API endpoints (not part of production)
    print("Now testing the API with /ping endpoint...")
    # simple test; in practice use an external tool like curl or Postman
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/ping")
    print(response.json())