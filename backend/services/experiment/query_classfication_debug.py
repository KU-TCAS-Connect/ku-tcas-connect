from services.llm_question_classification import QueryClassification
import os

query_classification = QueryClassification()
queries = [
    # "วิศวะซอฟต์แวร์และความรู้ รอบ1/1 นานาชาติ ภาคนานาชาติ มีเกณฑ์อะไรบ้าง",
    # "อยากทราบรายละเอียด",
    # "ราคาน้ำมันวันนี้เท่าไหร่",
    # "กระบวนการรับสมัครเป็นอย่างไร",
    # "มีกี่รอบ",
    # "วันปิดรับสมัครคือเมื่อไหร่",
    # "สมัครอย่างไร",
    # "ต้องใช้เอกสารอะไรบ้าง"
    # "มีเกณฑ์คุณสมบัติอะไรบ้าง",
    # "ต้องได้คะแนนขั้นต่ำเท่าไหร่",
    # "ต้องใช้วิชาอะไรบ้าง",
    "ต้องใช้ GPA เท่าไหร่",
    # "ต้องสอบเข้าไหม"
]

category = None

for query in queries:
    category = query_classification.classify(query)
    print(f"คำถาม: {query} → จัดหมวดหมู่เป็น: {category}")

category = category.intent

os.environ["QUERY"] = query  # Set environment variable

if category == "general_info":
    print("Executing search_txt.py")
    os.system("python search_txt.py")
elif category == "admission_criteria":
    print("Executing search_csv.py")
    os.system("python search_csv.py")
