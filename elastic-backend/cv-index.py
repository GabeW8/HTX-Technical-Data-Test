#!/usr/bin/env python3
import csv
import json
import requests

# Elasticsearch URLs for a 2-node cluster
ES_NODE_1 = "http://localhost:9200"
ES_NODE_2 = "http://elasticsearch-2:9200"  # Fallback node

INDEX_NAME = "cv-transcriptions"
# Updated CSV file path as per the test document requirement
CSV_FILE_PATH = "audio_samples/cs-valid-dev.csv"

def create_index():
    """Ensure the Elasticsearch index exists before inserting data."""
    index_settings = {
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1
        },
        "mappings": {
            "properties": {
                "generated_text": {"type": "text"},
                "duration": {"type": "float"},
                "age": {"type": "integer"},
                "gender": {"type": "keyword"},
                "accent": {"type": "keyword"}
            }
        }
    }
    
    url_primary = f"{ES_NODE_1}/{INDEX_NAME}"
    response = requests.put(url_primary, json=index_settings)
    if response.status_code in (200, 400):
        print("✅ Index verified or already exists on primary node.")
    else:
        print(f"❌ Failed to create index on primary node: {response.text}")
        # Fallback to secondary node
        url_secondary = f"{ES_NODE_2}/{INDEX_NAME}"
        response = requests.put(url_secondary, json=index_settings)
        if response.status_code in (200, 400):
            print("✅ Index verified or already exists on secondary node.")
        else:
            print(f"❌ Failed to create index on secondary node: {response.text}")

def bulk_index_data():
    """Reads CSV file and indexes it into Elasticsearch."""
    headers = {"Content-Type": "application/x-ndjson"}
    bulk_data = ""

    with open(CSV_FILE_PATH, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Create bulk action and data lines for Elasticsearch
            action = json.dumps({"index": {"_index": INDEX_NAME}})
            try:
                duration = float(row["duration"])
            except ValueError:
                duration = 0.0
            try:
                age = int(row["age"]) if row["age"].isdigit() else 0
            except ValueError:
                age = 0
            data = json.dumps({
                "generated_text": row["generated_text"],
                "duration": duration,
                "age": age,
                "gender": row["gender"],
                "accent": row["accent"]
            })
            bulk_data += f"{action}\n{data}\n"

    url_bulk_primary = f"{ES_NODE_1}/{INDEX_NAME}/_bulk"
    response = requests.post(url_bulk_primary, headers=headers, data=bulk_data)
    
    if response.status_code == 200:
        print("✅ Data indexed successfully on primary node!")
    else:
        print(f"❌ Failed to index data on primary node: {response.text}\nTrying secondary node...")
        url_bulk_secondary = f"{ES_NODE_2}/{INDEX_NAME}/_bulk"
        response = requests.post(url_bulk_secondary, headers=headers, data=bulk_data)
        if response.status_code == 200:
            print("✅ Data indexed successfully on secondary node!")
        else:
            print(f"❌ Failed to index data on secondary node: {response.text}")

if __name__ == "__main__":
    create_index()
    bulk_index_data()
