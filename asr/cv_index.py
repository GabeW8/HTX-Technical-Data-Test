import csv
import json
import requests

# Elasticsearch URL
ELASTICSEARCH_URL = "http://localhost:9200/cv-transcriptions/_bulk"

# Path to the CSV file (modify if necessary)
CSV_FILE_PATH = "audio_samples/cv-valid-dev.csv"

def bulk_index_data():
    """Reads CSV file and indexes it into Elasticsearch."""
    headers = {"Content-Type": "application/x-ndjson"}
    bulk_data = ""

    with open(CSV_FILE_PATH, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert CSV row to Elasticsearch format
            action = json.dumps({"index": {"_index": "cv-transcriptions"}})
            data = json.dumps({
                "generated_text": row["generated_text"],
                "duration": float(row["duration"]),
                "age": int(row["age"]),
                "gender": row["gender"],
                "accent": row["accent"]
            })
            bulk_data += f"{action}\n{data}\n"

    # Send bulk request
    response = requests.post(ELASTICSEARCH_URL, headers=headers, data=bulk_data)
    
    if response.status_code == 200:
        print("✅ Data indexed successfully!")
    else:
        print(f"❌ Failed to index data: {response.text}")

if __name__ == "__main__":
    bulk_index_data()
