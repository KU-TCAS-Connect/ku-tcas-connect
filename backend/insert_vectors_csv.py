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

def process_and_insert_data(df):
    points = []
    for _, row in df.iterrows():
        admission_round = row.get('round', 'N/A')
        admission_program = row.get('program_type', 'N/A')
        content = row.get('content', 'N/A')

        # try:
        #     embedding = ast.literal_eval(row['ada_embedding'])
        #     if not isinstance(embedding, list):
        #         raise ValueError("Embedding is not a valid list")
        # except (ValueError, SyntaxError) as e:
        #     print(f"Error parsing embedding for row {row['สาขาวิชา']}: {e}")
        #     continue 

        # Use BGEM3 to generate dense and sparse vectors
        sentences_1 = [content]  # Use the content of the row for encoding

        output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)

        dense_vector = output_1['dense_vecs'][0]
        
        print(dense_vector)
        
        # Extract the lexical weights (this is your sparse representation)
        lexical_weights = output_1['lexical_weights'][0]
        # Convert the lexical weights into a dictionary (index: weight)
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

        # Prepare the sparse vector: list of indices and values for sparse vector
        indices = list(sparse_vector_dict.keys())  # Indices of the sparse vector
        values = [float(x) for x in list(sparse_vector_dict.values())]
        sparse_vector = dict(zip(indices, values))
        # Create the point with the embedding and sparse vector
        point = PointStruct(
            id=data["id"],
            vector={
                "": data["embedding"],  # Dense vector
                "keywords": vector_class.qdrant_model.SparseVector(  # Sparse vector with "keywords"
                    indices=indices,  # List of indices
                    values=values  # List of values
                ),
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

        print(data)
        print(point)
        print("-------------------------------------------------------")
        points.append(point)
    if points:
        client.upsert(
            collection_name=vector_class.col_setting.collection_name["csv"],
            points=points
        )
        print(f"Inserted {len(points)} records into Qdrant.")
    else:
        print("No records were inserted due to embedding failures.")

if __name__ == "__main__":
    create_collection_table = True # Change to true to create new table
    
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