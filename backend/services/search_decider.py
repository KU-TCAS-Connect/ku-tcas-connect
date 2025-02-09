import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

# Define category labels
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

category_embeddings = {}
for category, phrases in CATEGORIES.items():
    category_embeddings[category] = np.mean(
        bge_model.encode(phrases, return_dense=True)["dense_vecs"], axis=0
    )

def classify_question(question: str) -> str:
    """Classifies the question into 'general_info' or 'admission_criteria'."""
    # Compute the query embedding
    query_embedding = bge_model.encode([question], return_dense=True)["dense_vecs"][0]
    
    # Compute cosine similarity with each category
    similarities = {
        category: cosine_similarity([query_embedding], [embedding])[0][0]
        for category, embedding in category_embeddings.items()
    }
    
    # Get the best matching category
    best_category = max(similarities, key=similarities.get)
    
    return best_category

query = "วิศวะซอฟต์แวร์และความรู้ รอบ1/1 นานาชาติ ภาคนานาชาติ มีเกณฑ์อะไรบ้าง"
category = classify_question(query)
print(f"จัดหมวดหมู่เป็น: {category}")
