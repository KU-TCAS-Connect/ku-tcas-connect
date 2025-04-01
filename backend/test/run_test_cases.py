import requests
import csv

API_URL = "http://localhost:8000"
SESSION_API = f"{API_URL}/new-session"
QUERY_API = f"{API_URL}/rag-query"

def create_new_session():
    response = requests.get(SESSION_API)
    if response.status_code == 200:
        return response.json()["session_id"]
    else:
        raise Exception(f"Failed to create new session: {response.text}")

def call_rag_query(session_id, query):
    payload = {
        "session_id": session_id,
        "query": query
    }
    response = requests.post(QUERY_API, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Failed to get response: {response.text}")

def read_test_cases_from_csv(input_csv_file):
    test_cases = []
    with open(input_csv_file, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row:
                test_cases.append(row[0])
    return test_cases

def main(input_csv_file):
    test_cases = read_test_cases_from_csv(input_csv_file)

    for query in test_cases:
        print(f"Processing query: {query}")
        try:
            session_id = create_new_session()
            response = call_rag_query(session_id, query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error processing query '{query}': {e}")

if __name__ == "__main__":
    input_csv_file = "run_test_cases.csv"
    main(input_csv_file)
