from datetime import datetime
from qdrant_client.models import PointStruct
import pandas as pd
import os

from database.connectdb import VectorStore
from services.bge_embedding import FlagModel

vector_class = VectorStore()
client = vector_class.qdrant_client

flag_class = FlagModel()
model = flag_class.bge_model

def chunk_text(text, num_chunks=5):
    words = text.split()
    total_words = len(words)
    chunk_size = total_words // num_chunks  # Divide the total words evenly among the chunks
    chunks = []

    start = 0
    for i in range(num_chunks):
        end = start + chunk_size
        if i == num_chunks - 1:  # Ensure the last chunk includes all remaining words
            end = total_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end

    return chunks

def process_and_insert_data(df, admission_info):
    points = []
    for _, row in df.iterrows():
        content = row['content']  # Assuming 'content' is the column with text from your .txt files
        filename = row['filename']  # Add the filename to the row (assuming 'filename' is part of the DataFrame)

        # Extract the admission info from the passed dictionary
        admission_round = admission_info['admission_round']
        admission_program = admission_info['admission_program']
        reference = admission_info['reference']

        sentences_1 = [content] 
        # Use BGEM3 to generate dense and sparse vectors
        output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)

        # Use BGEM3 dense vector for embedding
        dense_vector = output_1['dense_vecs'][0]
        
        lexical_weights = output_1['lexical_weights'][0]
        sparse_vector_dict = {token: weight for token, weight in lexical_weights.items()}

        data = {
            "id": str(vector_class.uuid_from_time_with_index(datetime.now(), 0)),
            "metadata": {
                "admission_round": admission_round,
                "admission_program": admission_program,
                "reference": reference,
                "created_at": datetime.now().isoformat(),
                "filename": filename,  # Add the filename to the metadata
                # "chunk_number": i + 1,
            },
            "contents": content,
            "embedding": dense_vector,  # Use BGEM3 dense vector
        }

        indices = list(sparse_vector_dict.keys())
        values = [float(x) for x in sparse_vector_dict.values()]
        sparse_vector = dict(zip(indices, values))

        point = PointStruct(
            id=data["id"],
            vector={ 
                "": data["embedding"],  # Dense vector from BGEM3
                "keywords": vector_class.qdrant_model.SparseVector(
                    indices=indices,
                    values=values
                ),
            },
            payload={ 
                "admission_round": data["metadata"]["admission_round"],
                "admission_program": data["metadata"]["admission_program"],
                "reference": data["metadata"]["reference"],
                "created_at": data["metadata"]["created_at"],
                "filename": data["metadata"]["filename"],  # Add filename to the payload
                # "chunk_number": data["metadata"]["chunk_number"],  # Add chunk number to the payload
                "contents": data["contents"],
                "lexical_weights": sparse_vector,
            },
        )
        
        points.append(point)
    if points:
        client.upsert(
            collection_name=vector_class.col_setting.collection_name["txt"],
            points=points
        )
        print(f"Inserted {len(points)} records into Qdrant.")
    else:
        print("No records were inserted due to embedding failures.")

if __name__ == "__main__":
    create_collection_table = True # Change to true to create new table
    
    if create_collection_table:
        vector_class.create_collection(vector_class.col_setting.collection_name["txt"])

    admission_info_mapping = {
        '1-0-เรียนล่วงหน้า': {
            'admission_round': "1",
            'admission_program': "เรียนล่วงหน้า",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/11/11/68_TCAS1_AP_1.1_edit.pdf"
        },
        '1-1-นานาชาติและภาษาอังกฤษ': {
            'admission_round': "1",
            'admission_program': "นานาชาติและภาษาอังกฤษ",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/11/68-TCAS1-International_Program_1.1.pdf"
        },
        '1-2-โอลิมปิกวิชาการ': {
            'admission_round': "1",
            'admission_program': "โอลิมปิกวิชาการ",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/03/68_TCAS1_Oiympics_1.2.pdf"
        },
        '1-1-รับนักกีฬาดีเด่น': {
            'admission_round': "1",
            'admission_program': "รับนักกีฬาดีเด่น",
            'reference':"https://admission.ku.ac.th/media/announcements/2024/10/03/68-TCAS1-Sport_1.1.pdf"
        },
        '1-1-ช้างเผือก': {
            'admission_round': "1",
            'admission_program': "ช้างเผือก",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/02/68_TCAS1_White_Elephant_1.1.pdf"
        },
        '1-2-ช้างเผือก': {
            'admission_round': "1",
            'admission_program': "ช้างเผือก",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/07/68_TCAS1_White_Elephant_1.2.pdf"
        },
        '2-0-เพชรนนทรี': {
            'admission_round': "2",
            'admission_program': "เพชรนนทรี",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/12/26/68-TCAS2-Diamond_Nontri.pdf"
        },
        '2-0-นานาชาติและภาษาอังกฤษ': {
            'admission_round': "2",
            'admission_program': "นานาชาติและภาษาอังกฤษ",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/18/68-TCAS2-International_Program.pdf"
        },
        '2-0-MOU': {
            'admission_round': "2",
            'admission_program': "MOU",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/25/68-TCAS2-KU_MOU.pdf"
        },
        '2-0-ลูกพระพิรุณ': {
            'admission_round': "2",
            'admission_program': "ลูกพระพิรุณ",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/15/68-TCAS2-Pra_Pirun.pdf"
        },
        '2-0-โควต้า30จังหวัด': {
            'admission_round': "2",
            'admission_program': "โควตา30จังหวัด",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/15/68-TCAS2-Province_30.pdf"
        },
        '2-0-นักเรียนดีเด่นจากโรงเรียนสาธิตแห่งมหาวิทยาลัยเกษตรศาสตร์': {
            'admission_round': "2",
            'admission_program': "นักเรียนดีเด่นจากโรงเรียนสาธิตแห่งมหาวิทยาลัยเกษตรศาสตร์",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/15/68-TCAS2-Province_30.pdf"
        },
        '2-0-ผู้มีความสามารถทางกีฬา': {
            'admission_round': "2",
            'admission_program': "ผู้มีความสามารถทางกีฬา",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/11/11/68-TCAS2-Sport.pdf"
        },
        '3-0-Admission': {
            'admission_round': "3",
            'admission_program': "Admission",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/31/68-TCAS3-Admission_edit-1.pdf"
        }
    }
    
    folder_path = "data-txt-new"

    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            print(f"Filename: {filename}")
        
            # Split filename by '-'
            split_parts = filename.split('-')
            if len(filename.split('-')) == 4:
                round, sub_round, round_name, topic = filename.split('-')
                num_topic = topic[:2]
                name_topic = topic[2:-4]
                print(f"No.{num_topic} - {name_topic}")
                
                name_get_detail = f"{round}-{sub_round}-{round_name}"
                # print(name_get_detail)
                # print(admission_info_mapping[name_get_detail])
                
                admission_info = admission_info_mapping.get(name_get_detail)
                # print(admission_info)
                if admission_info:
                    with open(f"{folder_path}/{filename}", "r", encoding="utf-8") as f:
                        content = f.read()
                
                    # Assuming you want to process the content as part of the DataFrame
                    df = pd.DataFrame({'content': [content], 'filename': [name_get_detail]})
                    print(df)
                    process_and_insert_data(df, admission_info)
                    print(f"Already insert")
                else:
                    print(f"No admission info found for {filename}")

            else:
                print("File name error (not equal to 4)")
            # print("Split parts:", split_parts)
            print("-----------------------------------")
            