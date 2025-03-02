import os
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging
import time

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Constants for file paths and API endpoint
AUDIO_DIR = "audio_samples/cv-valid-dev"
CSV_FILE = "audio_samples/cv-valid-dev.csv"
ASR_API_URL = os.getenv("ASR_API_URL", "http://localhost:8001/asr")  # Configurable API URL
MAX_WORKERS = 4
MAX_RETRIES = 3  # Number of retries before skipping a file

def process_file(index, filename):
    """Sends an audio file to the ASR API and updates the transcription."""
    file_path = os.path.join(AUDIO_DIR, filename)

    if not os.path.exists(file_path):
        logging.warning(f"Skipping missing file: {file_path}")
        return index, None

    retries = 0
    while retries < MAX_RETRIES:
        try:
            with open(file_path, "rb") as f:
                response = requests.post(ASR_API_URL, files={"file": f})
            
            if response.status_code == 200:
                result = response.json()
                transcription = result.get("transcription", "").strip()

                if not transcription:
                    logging.warning(f"Empty transcription for {file_path}")

                os.remove(file_path)  # Delete after successful transcription
                logging.info(f"Processed and deleted: {file_path}")
                return index, transcription
            else:
                logging.error(f"Error processing {file_path}: {response.text}")
        
        except Exception as e:
            logging.error(f"Exception processing {file_path}: {str(e)}")

        retries += 1
        logging.info(f"Retrying ({retries}/{MAX_RETRIES})...")
        time.sleep(2)

    return index, None

def main():
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        logging.error(f"Failed to read CSV file: {str(e)}")
        return

    if "generated_text" not in df.columns:
        df["generated_text"] = ""

    unprocessed = df[df["generated_text"].isna() | (df["generated_text"] == "")]
    if unprocessed.empty:
        logging.info("All files transcribed. Exiting.")
        return

    rows = list(unprocessed[["filename"]].itertuples(index=True, name=None))

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in tqdm(executor.map(lambda r: process_file(*r), rows), total=len(rows), desc="Processing audio files"):
            results.append(result)

    for index, transcription in results:
        if transcription is not None:
            df.at[index, "generated_text"] = transcription

    try:
        df.to_csv(CSV_FILE, index=False, mode='w', encoding='utf-8')
        logging.info("CSV updated with transcriptions.")
    except Exception as e:
        logging.error(f"Failed to save CSV: {str(e)}")

if __name__ == "__main__":
    main()
