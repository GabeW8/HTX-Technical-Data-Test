import os
import io
import torch
import soundfile as sf
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

app = FastAPI()

# Load the ASR model
model_name = os.getenv("MODEL_NAME", "facebook/wav2vec2-large-960h")
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Maximum allowed file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

@app.get("/ping")
def ping():
    """Health check endpoint"""
    return {"message": "pong"}

@app.post("/asr")
async def transcribe_audio(file: UploadFile = File(...)):
    """Process an uploaded MP3 file and return its transcription and duration."""
    try:
        # Ensure file is an MP3
        if not file.filename.endswith(".mp3"):
            raise HTTPException(status_code=400, detail="Only MP3 files are supported.")

        # Save uploaded file temporarily
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as temp_file:
            temp_file.write(await file.read())

        # Read the saved file
        audio_data, sample_rate = sf.read(temp_filename)

        # Convert stereo to mono if necessary
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample audio to 16kHz if needed
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        # Calculate duration
        duration = round(len(audio_data) / sample_rate, 2)

        # Prepare input and run inference
        input_values = processor(audio_data, return_tensors="pt", sampling_rate=16000).input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        # Delete processed file
        os.remove(temp_filename)

        return {"transcription": transcription, "duration": str(duration)}

    except Exception as e:
        # Delete file if it was saved before an error occurred
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return {"error": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
