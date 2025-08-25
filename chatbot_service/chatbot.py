import os
import json
import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))

INDEX_FILE = '/home/it/Assignment/shared_data/vector_index.faiss'
METADATA_FILE = '/home/it/Assignment/shared_data/metadata.json'
CHAT_HISTORY_FILE = '/home/it/Assignment/shared_data/chats.json'
EMBEDDING_DIM = 768

app = Flask(__name__,
            template_folder='/home/it/Assignment/chatbot_service/templates',
            static_folder='static')


# ---------------- FAISS / Metadata -----------------

def load_faiss_index():
    try:
        if os.path.exists(INDEX_FILE):
            return faiss.read_index(INDEX_FILE)
        else:
            return faiss.IndexFlatL2(EMBEDDING_DIM)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return faiss.IndexFlatL2(EMBEDDING_DIM)


def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def generate_text_embedding(text: str):
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text
        )
        embedding = response['embedding']
        print(f"Generated embedding of length {len(embedding)}")
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)


# ---------------- Chat History Helpers -----------------

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)


# ---------------- Routes -----------------

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        query_text = data.get('query', '').strip()
        chat_id = data.get('chat_id')  # existing chat id if continuing

        print("Received query:", query_text)
        if not query_text:
            return jsonify({'answer': 'Please provide a valid query.'})

        history = load_chat_history()

        # If no chat_id, create a new one
        if not chat_id:
            chat_id = str(uuid4())
            history[chat_id] = {
                "title": query_text,  # use first user message
                "messages": []
            }

        # ----- Fetch context -----
        index = load_faiss_index()
        print("FAISS index loaded")
        metadata = load_metadata()
        print(f"Metadata loaded, entries: {len(metadata)}")

        query_embedding = generate_text_embedding(query_text)
        if query_embedding is None or len(query_embedding) != EMBEDDING_DIM:
            raise ValueError("Invalid embedding generated")

        k = 3
        D, I = index.search(np.expand_dims(query_embedding, axis=0), k)
        print(f"Search results indices: {I}, distances: {D}")

        retrieved_texts = []
        for idx in I[0]:
            if str(idx) in metadata:
                retrieved_texts.append(metadata[str(idx)]['text'])

        # ----- Generate answer -----
        if not retrieved_texts:
            answer = "Sorry, no relevant information found."
        else:
            context = "\n\n".join(retrieved_texts)
            prompt = (
                "Answer the question based ONLY on the following context. "
                "If you cannot answer from the context, say 'I don't have enough information.'\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query_text}\n\n"
                "Answer:"
            )

            response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
                prompt,
                generation_config={
                    "temperature": 0.5,
                    "max_output_tokens": 512,
                    "top_p": 0.95,
                    "top_k": 40
                }
            )

            answer = response.text.strip() if response.text else "I couldn't generate an answer."

        # ----- Save chat -----
        history[chat_id]["messages"].append({"user": query_text, "bot": answer})
        save_chat_history(history)

        print("Received answer from Gemini")
        return jsonify({'answer': answer, 'chat_id': chat_id})

    except Exception as e:
        print(f"Error in /chat: {e}", flush=True)
        return jsonify({'answer': 'An error occurred processing your query.'})


@app.route('/history', methods=['GET'])
def get_history():
    history = load_chat_history()
    summary = [{"chat_id": cid, "title": h["title"]} for cid, h in history.items()]
    return jsonify(summary)


@app.route('/chat/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    history = load_chat_history()
    if chat_id in history:
        return jsonify(history[chat_id])
    return jsonify({'error': 'Chat not found'}), 404


# ---------------- Run -----------------

if __name__ == '__main__':
    if not os.getenv("API_KEY"):
        print("WARNING: Missing Gemini API_KEY in environment.")
    app.run(debug=True)
