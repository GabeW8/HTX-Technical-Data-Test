import os
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging

# configure logging for debugging; set level to DEBUG to see more detailed output
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

# constants for file paths and API endpoint
AUDIO_DIR = "audio_samples/cv-valid-dev"  # directory containing MP3 files
CSV_FILE = "audio_samples/cv-valid-dev.csv"  # CSV file with filenames and metadata
ASR_API_URL = "http://localhost:8001/asr"  # endpoint for the ASR API
MAX_WORKERS = 4  # limit concurrency to avoid API overload

def process_file(index, filename):
    """
    Sends an audio file to the ASR API and returns its transcription.
    Deletes the file after successful processing.
    """
    file_path = os.path.join(AUDIO_DIR, filename)
    logging.debug(f"Processing file: {file_path}")
    
    if not os.path.exists(file_path):
        logging.warning(f"Skipping missing file: {file_path}")
        return index, None
    
    try:
        with open(file_path, "rb") as f:
            response = requests.post(ASR_API_URL, files={"file": f})
        
        logging.debug(f"API response for {filename}: {response.status_code} {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            transcription = result.get("transcription", "").strip()
            if not transcription:
                logging.warning(f"Empty transcription returned for {file_path}")
            os.remove(file_path)
            logging.info(f"Processed and deleted: {file_path}")
            return index, transcription
        else:
            logging.error(f"Error processing {file_path}: {response.text}")
            return index, None
    except Exception as e:
        logging.error(f"Exception processing {file_path}: {str(e)}")
        return index, None

def main():
    try:
        df = pd.read_csv(CSV_FILE)
        logging.debug(f"CSV loaded with {len(df)} rows. Columns: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Failed to read CSV file: {str(e)}")
        return

    if "generated_text" not in df.columns:
        df["generated_text"] = ""
        logging.debug("Created 'generated_text' column in CSV")
    
    unprocessed = df[df["generated_text"].isna() | (df["generated_text"] == "")]
    logging.info(f"Found {len(unprocessed)} unprocessed rows")
    if unprocessed.empty:
        logging.info("All files have been transcribed already. Exiting.")
        return

    rows = list(unprocessed[["filename"]].itertuples(index=True, name=None))

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in tqdm(executor.map(lambda r: process_file(*r), rows), total=len(rows), desc="Processing audio files"):
            results.append(result)

    for index, transcription in results:
        if transcription is not None:
            df.at[index, "generated_text"] = transcription
            logging.debug(f"Updated row {index}: {df.at[index, 'generated_text']}")
    
    try:
        df.to_csv(CSV_FILE, index=False, mode='w', encoding='utf-8')
        logging.info("Transcription completed. CSV updated.")
        
        # Verify file write by reloading and checking
        df2 = pd.read_csv(CSV_FILE)
        logging.debug("First 10 rows of updated CSV:")
        print(df2[["filename", "generated_text"]].head(10))
    except Exception as e:
        logging.error(f"Failed to save CSV: {str(e)}")

if __name__ == "__main__":
    main()
