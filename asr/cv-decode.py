# cv-decode.py
import os
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

AUDIO_DIR: str = "audio_samples/cv-valid-dev"       # Directory containing MP3 files
CSV_FILE: str = "audio_samples/cv-valid-dev.csv"      # CSV file with filenames and metadata
ASR_API_URL: str = "http://localhost:8001/asr"        # Endpoint for the ASR API
MAX_WORKERS: int = 4                                  # Limit concurrency to avoid API overload
BATCH_SIZE: int = 50                                  # Update CSV every 50 files
FORCE_REPROCESS: bool = True                          # Process every row regardless of existing values

def process_file(index: int, filename: str) -> Tuple[int, Optional[str], Optional[str], str]:
    """
    Sends an audio file to the ASR API and returns its transcription, duration, and processing status.
    Deletes the file after successful processing.
    
    Returns:
        (index, transcription, duration, processing_status)
    """
    file_path = os.path.join(AUDIO_DIR, filename)
    logging.debug(f"Processing file: {file_path}")
    
    if not os.path.exists(file_path):
        logging.warning(f"Skipping missing file: {file_path}")
        return index, None, None, "MISSING"
    
    try:
        with open(file_path, "rb") as f:
            response = requests.post(ASR_API_URL, files={"file": f})
        
        logging.debug(f"API response for {filename}: {response.status_code} {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            transcription = result.get("transcription", "").strip()
            duration = result.get("duration", "")
            status = "SUCCESS" if transcription else "FAILED"
            if not transcription:
                logging.warning(f"Empty transcription returned for {file_path}")
            os.remove(file_path)
            logging.info(f"Processed and deleted: {file_path}")
            return index, transcription, duration, status
        else:
            logging.error(f"Error processing {file_path}: {response.text}")
            return index, None, None, "FAILED"
    except Exception as e:
        logging.error(f"Exception processing {file_path}: {str(e)}")
        return index, None, None, "FAILED"

def save_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    Saves the DataFrame to CSV and logs success or failure.
    """
    try:
        df.to_csv(file_path, index=False, encoding="utf-8")
        logging.info(f"CSV updated and saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save CSV: {str(e)}")

def filter_unprocessed_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns rows that need processing.
    If FORCE_REPROCESS is True, returns the entire DataFrame.
    Otherwise, returns rows with empty or NaN 'generated_text' or status marked as 'FAILED'.
    """
    if FORCE_REPROCESS:
        return df
    else:
        return df[
            df["generated_text"].isna() | (df["generated_text"] == "") | (df["processing_status"] == "FAILED")
        ]

def main() -> None:
    try:
        df = pd.read_csv(CSV_FILE)
        logging.debug(f"CSV loaded with {len(df)} rows. Columns: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Failed to read CSV file: {str(e)}")
        return

    if "generated_text" not in df.columns:
        df["generated_text"] = ""
        logging.debug("Created 'generated_text' column in CSV")
    if "duration" not in df.columns:
        df["duration"] = ""
        logging.debug("Created 'duration' column in CSV")
    if "processing_status" not in df.columns:
        df["processing_status"] = ""
        logging.debug("Created 'processing_status' column in CSV")
    
    unprocessed_df = filter_unprocessed_rows(df)
    logging.info(f"Found {len(unprocessed_df)} rows to process")
    rows = list(unprocessed_df[["filename"]].itertuples(index=True, name=None))
    
    processed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in tqdm(
            executor.map(lambda r: process_file(*r), rows),
            total=len(rows),
            desc="Processing audio files"
        ):
            index, transcription, duration, status = result
            if transcription is not None:
                df.at[index, "generated_text"] = transcription
                df.at[index, "duration"] = duration
                df.at[index, "processing_status"] = status
                logging.debug(f"Updated row {index}: {transcription}, duration: {duration}, status: {status}")
            else:
                df.at[index, "processing_status"] = status
                logging.debug(f"Updated row {index} with status: {status}")
            processed_count += 1
            
            if processed_count % BATCH_SIZE == 0:
                save_csv(df, CSV_FILE)
    
    save_csv(df, CSV_FILE)
    try:
        df2 = pd.read_csv(CSV_FILE)
        logging.debug("First 10 rows of updated CSV:")
        print(df2[["filename", "generated_text", "duration", "processing_status"]].head(10))
    except Exception as e:
        logging.error(f"Failed to verify CSV: {str(e)}")

if __name__ == "__main__":
    main()
