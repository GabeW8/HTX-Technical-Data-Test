import os
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor  #### Switched to ThreadPoolExecutor instead of ProcessPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

AUDIO_DIR = "audio_samples/cv-valid-dev"
CSV_FILE = "audio_samples/cv-valid-dev.csv"
ASR_API_URL = "http://localhost:8001/asr"

def process_file(row):
    """
    Process a single audio file:
      - Calls the ASR API to get the transcription.
      - Deletes the file upon successful processing.
    
    Returns a tuple: (index, transcription, error_message)
    """
    index, filename = row  # row is a tuple: (index, filename)
    file_path = os.path.join(AUDIO_DIR, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        msg = f"File not found: {file_path}"
        logging.warning(msg)
        return (index, None, msg)
    
    try:
        with open(file_path, "rb") as f:
            response = requests.post(ASR_API_URL, files={"file": f})

        if response.status_code == 200:
            result = response.json()
            transcription = result.get("transcription", "")
            # Delete the file after successful processing
            os.remove(file_path)
            logging.info(f"Processed and deleted: {file_path}")
            return (index, transcription, None)
        else:
            error_msg = f"Error {response.status_code}: {response.text}"
            logging.error(f"{file_path} - {error_msg}")
            return (index, None, error_msg)
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Exception processing {file_path}: {error_msg}")
        return (index, None, error_msg)

def main():
    df = pd.read_csv(CSV_FILE)

    if "generated_text" not in df.columns:
        df["generated_text"] = ""

    rows = list(df[["filename"]].itertuples(index=True, name=None))

    #### 🔹 OPTION 1: SEQUENTIAL PROCESSING (Most Stable) ####
    # results = []
    # for row in tqdm(rows, desc="Processing audio files"):
    #     result = process_file(row)
    #     results.append(result)

    #### 🔹 OPTION 2: THREAD-BASED CONCURRENCY (Use if Needed) ####
    # Use ThreadPoolExecutor (safe for I/O operations)
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:  #### Reduced concurrency to avoid API overload
        for result in tqdm(executor.map(process_file, rows), total=len(rows), desc="Processing audio files"):
            results.append(result)
    

    for index, transcription, error in results:
        if transcription is not None:
            df.at[index, "generated_text"] = transcription

    df.to_csv(CSV_FILE, index=False)
    logging.info("✅ Transcription completed. CSV updated.")

if __name__ == "__main__":
    main()
