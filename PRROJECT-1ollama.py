from neo4j import GraphDatabase
from flask import Flask, request
from linebot import LineBotApi
from linebot.models import TextSendMessage
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import requests
from neo4j.exceptions import ServiceUnavailable
from concurrent.futures import ThreadPoolExecutor
import re

# Neo4j settings
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "Lutfee2salaeh")

# LINE Bot settings
LINE_CHANNEL_ACCESS_TOKEN = 'BRf2b3KJxrhuUH3bTYIQ/TfCnVayWkAhBsxahSMftqGCsPHX70i5Rba0O8JZqiPAgGZqHTn+bjDoKL69BJcVnZksNUwbvev8DoHot9RGPEGzs5KPj73qAgA+O0pmqqpRP0rybmBB4BKECdpGafh+bAdB04t89/1O/w1cDnyilFU='

# Ollama API settings
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Connect to Neo4j
try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
except ServiceUnavailable as e:
    print(f"Failed to connect to Neo4j: {e}")

# Use SentenceTransformer model
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# Query to fetch data from Neo4j
cypher_query = '''
MATCH (n)
WHERE n:Greeting OR n:Question
RETURN n.name as name, n.msg_reply as reply, labels(n)[0] as type
'''

corpus = []
responses = {}
node_types = {}

# Fetch data from Neo4j
with driver.session() as session:
    results = session.run(cypher_query)
    for record in results:
        name = record['name']
        reply = record['reply']
        if name and reply:  # Check if name and reply are not None
            corpus.append(name)
            responses[name] = reply
            node_types[name] = record['type']
            print(f"Loaded from Neo4j: {name} - {record['type']}")

print(f"Total items loaded from Neo4j: {len(corpus)}")

# Convert all texts in corpus to vectors
if corpus:  # Ensure corpus is not empty
    corpus_vec = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()
else:
    print("No valid items in corpus to encode.")



# FAISS index setup
vector_dimension = corpus_vec.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
index.add(np.array(corpus_vec))

# Set up ThreadPoolExecutor for parallel Ollama calls
executor = ThreadPoolExecutor(max_workers=4)

# Conversation history
conversation_history = {}
session = requests.Session()

def is_badminton_related(sentence):
    badminton_keywords = ["แบดมินตัน", "แร็กเก็ต", "ลูกขนไก่", "คอร์ต", "เน็ต", "สมาคมแบดมินตัน", "โอลิมปิก", "เซตต์", "แมทช์", "เกม", "แต้ม", "เซิร์ฟ", "ตบ", "ดรอป", "เสมอ", "ชนะ", "แพ้", "ทัวร์นาเมนต์", "แชมป์", "อันดับโลก", "สายการแข่งขัน"]
    return any(keyword in sentence for keyword in badminton_keywords)

def query_ollama(prompt, is_rephrase=False, user_id=None, is_badminton=False):
    history = conversation_history.get(user_id, [])
    context = "\n".join([f"Human: {h['human']}\nAI: {h['ai']}" for h in history[-2:]])  # Last 2 conversation
    
    if is_rephrase:
        instruction = (f"เรียบเรียงคำตอบต่อไปนี้ใหม่โดยรักษาความหมายเดิม ไม่เพิ่มข้อมูลใหม่: {prompt} "
                       f"ตอบเป็นภาษาไทย กระชับ ตรงประเด็น ไม่เกิน 2 ประโยค")
    else:
        if is_badminton:
            instruction = (f"คุณเป็นผู้เชี่ยวชาญด้านแบดมินตัน ตอบคำถามนี้เกี่ยวกับแบดมินตันเท่านั้น: {prompt} "
                           f"บริบทการสนทนา:\n{context}\nตอบเป็นภาษาไทย กระชับ ตรงประเด็น ไม่เกิน 2 ประโยค")
        else:
            instruction = (f"ตอบคำถามนี้โดยคำนึงถึงบริบทการสนทนา:\n{context}\n\nคำถาม: {prompt}\n"
                           f"ตอบเป็นภาษาไทย กระชับ ตรงประเด็น ไม่เกิน 2 ประโยค")

    payload = {
        "model": "supachai/llama-3-typhoon-v1.5",
        "prompt": instruction,
        "max_tokens": 30,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json"
    }

    # try:
    #     response = requests.post(OLLAMA_API_URL, headers=headers, json=payload, timeout=10)
    #     response.raise_for_status()
    #     data = response.json()
    #     thai_response = data.get("response", "").strip()
        
    #     # Filter only Thai text and some special characters
    #     thai_only = re.sub(r'^[^\u0E00-\u0E7F]+', '', re.sub(r'[^ก-๙A-Za-z0-9\s!@#$%^&*(),.?":{}|<>-]', '', thai_response))
    #     thai_only = thai_only.strip()
        
    #     return thai_only if thai_only else "ขออภัยครับ ไม่สามารถตอบคำถามได้"
    # except requests.RequestException as e:
    #     print(f"Failed to get a response from Ollama: {e}")
    #     return "ขอโทษครับ มีปัญหาในการตอบคำถาม"
    response = session.post(OLLAMA_API_URL, headers=headers, data= json.dumps(payload))
    if response.status_code == 200:
        response_data = response.text
        data = json.loads(response_data)
        decoded_text = data["response"] 
        return  decoded_text 
       
    else:
        return (f"Failed to get a response: {response.status_code}, {response.text}")

def save_to_neo4j(question, answer):
    global last_added_question
    with driver.session() as session:
        result = session.run("MATCH (q:Question {name: $question}) RETURN q", question=question)
        if not result.single():
            session.run(
                "CREATE (q:Question {name: $question, msg_reply: $answer})",
                question=question, answer=answer
            )
            corpus.append(question)
            responses[question] = answer
            node_types[question] = 'Question'
            new_vector = model.encode([question], convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()
            faiss.normalize_L2(new_vector)
            index.add(new_vector)
            print(f"New question added to Neo4j: {question}")
            last_added_question = question  # เก็บคำถามล่าสุดที่เพิ่ม
        else:
            print(f"Question already exists in Neo4j: {question}")

def delete_last_added_question():
    global last_added_question
    if last_added_question:
        with driver.session() as session:
            session.run("MATCH (q:Question {name: $question}) DELETE q", question=last_added_question)
            print(f"Deleted the last added question: {last_added_question}")
            last_added_question = None  # รีเซ็ตคำถามล่าสุดหลังจากลบ
    else:
        print("No question to delete.")

        
def compute_response(sentence, user_id):
    search_vector = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy().reshape(1, -1)
    faiss.normalize_L2(search_vector)

    distances, indices = index.search(search_vector, 3)  # Fetch top 3 results
    similarity_scores = 1 - distances[0]
    
    print(f"Query: {sentence}")
    print(f"Top 3 similarity scores: {similarity_scores}")
    print(f"Top 3 matching items: {[corpus[i] for i in indices[0]]}")
    
    is_badminton = is_badminton_related(sentence)

    if similarity_scores[0] >= 0.7:  # Lowered threshold for High confidence match
        match_greeting = corpus[indices[0][0]]
        original_response = responses.get(match_greeting, "ขอโทษครับ ไม่มีข้อมูลเกี่ยวกับคำถามนี้")
        node_type = node_types.get(match_greeting, "Unknown")
        print(f"Original response from Neo4j ({node_type}): {similarity_scores[0]},{original_response}")

        # Fast response for high confidence matches
        if similarity_scores[0] >= 0.9:  # Very high confidence
            return f"(Neo4j): {original_response}"
        
        # Use a simpler rephrasing for slightly lower confidence
        response_msg = query_ollama(f"พูดว่า: {original_response}", is_rephrase=True, user_id=user_id, is_badminton=is_badminton)
        return f"(Neo4j + Ollama): {response_msg}"
    
    elif similarity_scores[0] >= 0.6:
        context = " ".join([responses.get(corpus[idx], "") for idx in indices[0]])
        prompt = f"คำถาม: {sentence}\n\nข้อมูลที่เกี่ยวข้อง: {context}\n\nกรุณาตอบคำถามโดยใช้ข้อมูลที่ให้มา"
        response_msg = query_ollama(prompt, user_id=user_id, is_badminton=is_badminton)
        response_msg = f"(Neo4j + Ollama): {response_msg}"
    else:
        print("ไม่มีคำตอบใน Neo4j กำลังสร้างคำตอบใหม่จาก Ollama รอสักครู่...")
        response_msg = query_ollama(sentence, user_id=user_id, is_badminton=is_badminton)
        response_msg = f"{response_msg}"
    
    if similarity_scores[0] < 0.6:
        save_to_neo4j(sentence, response_msg)

    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append({"human": sentence, "ai": response_msg})
    
    return response_msg


app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = 'BRf2b3KJxrhuUH3bTYIQ/TfCnVayWkAhBsxahSMftqGCsPHX70i5Rba0O8JZqiPAgGZqHTn+bjDoKL69BJcVnZksNUwbvev8DoHot9RGPEGzs5KPj73qAgA+O0pmqqpRP0rybmBB4BKECdpGafh+bAdB04t89/1O/w1cDnyilFU='

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    print(f"Received body: {body}")  # เพิ่มบรรทัดนี้เพื่อตรวจสอบ payload ที่เข้ามาา

    try:
        json_data = json.loads(body)

        if 'events' in json_data and len(json_data['events']) > 0:
            event = json_data['events'][0]
            msg = event['message']['text']
            tk = event['replyToken']
            user_id = event['source']['userId']
            
            response_msg = compute_response(msg, user_id)  # ฟังก์ชันนี้ใช้สำหรับคอมพิวเตอร์ตอบกลับ
            line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
            line_bot_api.reply_message(tk, TextSendMessage(text=response_msg))
            print(msg, tk)
        else:
            print("No events found in the request.")
    
    except Exception as e:
        print(f"Error: {e}")
        print(body)
    
    return 'OK'


if __name__ == '__main__':
    app.run(port=5201, threaded=True)
