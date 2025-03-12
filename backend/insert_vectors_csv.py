from datetime import datetime
import pandas as pd
from qdrant_client.models import PointStruct
import ast

from database.connectdb import VectorStore
from services.bge_embedding import FlagModel

vector_class = VectorStore()
client = vector_class.qdrant_client

flag_class = FlagModel()
model = flag_class.bge_model

def read_csv_data(file_path):
    df = pd.read_csv(file_path)
    return df

def process_and_insert_data(df, batch_size=5):
    points = []
    for _, row in df.iterrows():
        admission_round = row.get('round', 'N/A')
        admission_program = row.get('program_type', 'N/A')
        content = row.get('content', 'N/A')

        sentences_1 = [content]  # Use the content of the row for encoding
        output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=True)

        dense_vector = output_1['dense_vecs'][0]
        colbert = output_1['colbert_vecs'][0]
        lexical_weights = output_1['lexical_weights'][0]
        sparse_vector_dict = {token: weight for token, weight in lexical_weights.items()}

        data = {
            "id": str(vector_class.uuid_from_time(datetime.now())),
            "metadata": {
                "major": row['สาขาวิชา'],
                "admission_round": admission_round,
                "admission_program": admission_program,
                "reference": row['แหล่งที่มา'],
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": dense_vector,
        }

        indices = list(sparse_vector_dict.keys())  # Indices of the sparse vector
        values = [float(x) for x in list(sparse_vector_dict.values())]
        sparse_vector = dict(zip(indices, values))

        point = PointStruct(
            id=data["id"],
            vector={
                "bge-dense": data["embedding"],  # Dense vector
                "keywords": vector_class.qdrant_model.SparseVector(  # Sparse vector with "keywords"
                    indices=indices,  # List of indices
                    values=values  # List of values
                ),
                "colbert": colbert,
            },
            payload={
                "major": data["metadata"]["major"],
                "admission_round": data["metadata"]["admission_round"],
                "admission_program": data["metadata"]["admission_program"],
                "reference": data["metadata"]["reference"],
                "created_at": data["metadata"]["created_at"],
                "contents": data["contents"],
                "lexical_weights": sparse_vector,  # Store sparse vector in payload
            },
        )

        points.append(point)
        
        # If the batch size is reached, insert and clear the list
        if len(points) >= batch_size:
            client.upsert(
                collection_name=vector_class.col_setting.collection_name["csv"],
                points=points
            )
            print(f"Inserted {len(points)} records into Qdrant.")
            points = []  # Clear points for the next batch

    # Insert any remaining points after the loop
    if points:
        client.upsert(
            collection_name=vector_class.col_setting.collection_name["csv"],
            points=points
        )
        print(f"Inserted {len(points)} records into Qdrant.")

if __name__ == "__main__":
    create_collection_table = True  # Change to true to create new table
    
    if create_collection_table:
        vector_class.create_collection(vector_class.col_setting.collection_name["csv"])

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
        df = read_csv_data(f"data/{file}")
        process_and_insert_data(df)
