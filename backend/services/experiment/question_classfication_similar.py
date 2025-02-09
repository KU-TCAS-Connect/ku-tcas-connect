import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

CATEGORIES = {
    "ข้อมูลทั่วไป": [
        "กระบวนการรับสมัครเป็นอย่างไร?",
        "มีกี่รอบ?",
        "วันปิดรับสมัครคือเมื่อไหร่?",
        "สมัครอย่างไร?",
        "ต้องใช้เอกสารอะไรบ้าง?"
    ],
    "เกณฑ์การรับสมัคร": [
        "มีเกณฑ์คุณสมบัติอะไรบ้าง?",
        "ต้องได้คะแนนขั้นต่ำเท่าไหร่?",
        "ต้องใช้วิชาอะไรบ้าง?",
        "ต้องใช้ GPA เท่าไหร่?",
        "ต้องสอบเข้าไหม?"
    ]
}

def encode_text(text):

    output = bge_model.encode([text], return_dense=True, return_sparse=True)

    dense_vector = np.array(output["dense_vecs"][0])  # Dense Vector
    sparse_vector = output["lexical_weights"][0]  # Sparse Vector (Dictionary)

    return dense_vector, sparse_vector

def compute_sparse_similarity(vec1, vec2):
    common_keys = set(vec1.keys()) & set(vec2.keys())

    if not common_keys:
        return 0.0

    dot_product = sum(vec1[k] * vec2[k] for k in common_keys)
    norm1 = sum(v ** 2 for v in vec1.values()) ** 0.5
    norm2 = sum(v ** 2 for v in vec2.values()) ** 0.5

    return dot_product / (norm1 * norm2) if (norm1 * norm2) != 0 else 0.0

category_vectors = {
    category: [
        encode_text(text) for text in examples
    ]
    for category, examples in CATEGORIES.items()
}

DENSE_THRESHOLD = 0.5
SPARSE_THRESHOLD = 0.3

def classify_question(question: str) -> str:
    query_dense, query_sparse = encode_text(question)

    best_category = "ไม่เกี่ยวข้อง"
    best_score = 0

    for category, examples in category_vectors.items():
        dense_scores = [cosine_similarity([query_dense], [dense])[0][0] for dense, _ in examples]
        sparse_scores = [compute_sparse_similarity(query_sparse, sparse) for _, sparse in examples]

        avg_dense_score = np.mean(dense_scores)
        avg_sparse_score = np.mean(sparse_scores)

        combined_score = (avg_dense_score + avg_sparse_score) / 2  # รวม Dense และ Sparse

        if combined_score > best_score:
            best_category = category
            best_score = combined_score

    if best_score < min(DENSE_THRESHOLD, SPARSE_THRESHOLD):
        return "ไม่เกี่ยวข้อง"

    return best_category

# ตัวอย่างการใช้งาน
queries = [
    "วิศวะซอฟต์แวร์และความรู้ รอบ1/1 นานาชาติ ภาคนานาชาติ มีเกณฑ์อะไรบ้าง",
    "อยากทราบรายละเอียด",
    "ราคาน้ำมันวันนี้เท่าไหร่",
]

for query in queries:
    category = classify_question(query)
    print(f"คำถาม: {query} → จัดหมวดหมู่เป็น: {category}")
