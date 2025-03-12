import os
import pandas as pd
from datetime import datetime
import uuid
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Elasticsearch connection
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
es_client = Elasticsearch(hosts=[ES_HOST])

# Define Elasticsearch index name
INDEX_NAME = "ku_tcas_criteria"

# Function to create index with proper mappings
def create_es_index():
    if not es_client.indices.exists(index=INDEX_NAME):
        es_client.indices.create(
            index=INDEX_NAME,
            body={
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "thai_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "admission_round": {"type": "keyword"},
                        "admission_program": {"type": "text", "analyzer": "thai_analyzer"},
                        "contents": {"type": "text", "analyzer": "thai_analyzer"},
                        "reference": {"type": "keyword"},
                        "created_at": {"type": "date"}
                    }
                }
            }
        )
        print(f"Created index: {INDEX_NAME}")
    else:
        print(f"Index {INDEX_NAME} already exists.")

# Function to read CSV files
def read_csv_data(file_path):
    return pd.read_csv(file_path)

# Function to process and insert data into Elasticsearch
def process_and_insert_data(df):
    es_documents = []

    for _, row in df.iterrows():
        admission_round = row.get('round', 'N/A')
        admission_program = row.get('program_type', 'N/A')
        content = row.get('content', 'N/A')
        reference = row.get("แหล่งที่มา", "Unknown")

        # Create a document with a unique UUID as ID
        es_doc = {
            "_index": INDEX_NAME,
            "_id": str(uuid.uuid4()),  # Unique ID using UUID
            "_source": {
                "admission_round": admission_round,
                "admission_program": admission_program,
                "contents": content,
                "reference": reference,
                "created_at": datetime.now().isoformat(),
            }
        }
        es_documents.append(es_doc)

    # Bulk insert into Elasticsearch
    if es_documents:
        bulk(es_client, es_documents)
        print(f"Inserted {len(es_documents)} records into Elasticsearch.")

# Main script
if __name__ == "__main__":
    create_es_index()  # Ensure index exists

    # List of CSV files
    csv_list_file = [
        '1-0-เรียนล่วงหน้า.csv',
        '1-1-ช้างเผือก.csv',
        '1-1-นานาชาติและภาษาอังกฤษ.csv',
        '1-1-รับนักกีฬาดีเด่น.csv',
        '1-2-ช้างเผือก.csv',
        '1-2-โอลิมปิกวิชาการ.csv',
        '2-0-MOU.csv',
        '2-0-โควต้า30จังหวัด.csv',
        '2-0-เพชรนนทรี.csv',
        '2-0-ลูกพระพิรุณ.csv',
        '2-0-นานาชาติและภาษาอังกฤษ.csv',
        '2-0-ผู้มีความสามารถทางกีฬา.csv',
        '2-0-นักเรียนดีเด่นจากโรงเรียนสาธิตแห่งมหาวิทยาลัยเกษตรศาสตร์.csv',
        '3-0-Admission.csv',
    ]
    
    for file in csv_list_file:
        file_path = f"backend\\data\\{file}"  # Fixing escape sequence warning
        if os.path.exists(file_path):
            df = read_csv_data(file_path)
            process_and_insert_data(df)
        else:
            print(f"File not found: {file_path}")
