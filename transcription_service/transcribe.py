import os
import re
import time
import threading
import requests
import pickle
import faiss
import numpy as np
import json
from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# == CONFIGURATION ==
SERVICE_ACCOUNT_FILE = 'service_account.json'  # Service account key JSON file
SCOPES = ['https://www.googleapis.com/auth/drive']

# Gemini
genai.configure(api_key=os.getenv("API_KEY"))

# AssemblyAI
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# compute absolute path to templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR   = os.path.join(BASE_DIR, 'static')

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)


app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")

# GLOBAL VARIABLES
INDEX_FILE = '/home/it/Assignment/shared_data/vector_index.faiss'
METADATA_FILE = '/home/it/Assignment/shared_data/metadata.json'
EMBEDDING_DIM = 768  # For most Gemini embedding models
index_lock = threading.Lock()

processing_videos = set()
processing_videos_lock = threading.Lock()

# == Google Drive Service ==
def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build('drive', 'v3', credentials=creds)

# == Translation (Gemini) ==
def translate_with_gemini(text: str) -> str:
    prompt = (
        "Translate the text(Each and every line of text) given to you to English."
        "No need to give a number for each line. Give it as a whole paragraph from the beginning, nothing other than that.\n\n"
        f"Text:\n{text}"
    )
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return (response.text or "").strip()

# == Helper: Extract Folder ID ==
def extract_folder_id(input_str):
    match = re.search(r'/folders/([a-zA-Z0-9\_-]+)', input_str)
    if match:
        return match.group(1)
    else:
        return input_str.strip()

# == Google Drive API ==
def list_videos(video_folder_id):
    service = get_drive_service()
    if not video_folder_id:
        return []
    query = f"'{video_folder_id}' in parents and mimeType contains 'video/'"
    results = service.files().list(
        q=query,
        fields="files(id, name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute()
    return results.get('files', [])

def list_transcripts(transcript_folder_id):
    service = get_drive_service()
    if not transcript_folder_id:
        return set()
    transcript_names = set()
    page_token = None
    while True:
        response = service.files().list(
            q=f"'{transcript_folder_id}' in parents and mimeType='text/plain'",
            spaces='drive',
            fields='nextPageToken, files(name)',
            pageToken=page_token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True
        ).execute()
        for file in response.get('files', []):
            transcript_names.add(file['name'])
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return transcript_names

def download_video(file_id, target_path):
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    os.makedirs('tmp', exist_ok=True)
    with open(target_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return target_path

def upload_transcription(file_name, text_path, transcript_folder_id):
    service = get_drive_service()
    if not transcript_folder_id:
        return
    file_metadata = {
        'name': file_name,
        'parents': [transcript_folder_id],
        'mimeType': 'text/plain'
    }
    media = MediaFileUpload(text_path, mimetype='text/plain')
    service.files().create(
        body=file_metadata,
        media_body=media,
        supportsAllDrives=True
    ).execute()

# == FAISS functions (load, save, embed) - unchanged ==
def load_faiss_index():
    try:
        if os.path.exists(INDEX_FILE):
            return faiss.read_index(INDEX_FILE)
        else:
            return faiss.IndexFlatL2(EMBEDDING_DIM)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return faiss.IndexFlatL2(EMBEDDING_DIM)

def save_faiss_index(index):
    try:
        faiss.write_index(index, INDEX_FILE)
        print(f"FAISS index saved successfully to {INDEX_FILE}")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

def save_metadata(metadata):
    try:
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        print(f"Metadata saved successfully to {METADATA_FILE}")
    except Exception as e:
        print(f"Error saving metadata: {e}")

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

def store_transcript_embedding(file_name, translated_text):
    with index_lock:
        index = load_faiss_index()
        metadata = load_metadata()
        embedding = generate_text_embedding(translated_text)
        index.add(np.expand_dims(embedding, axis=0))
        vector_id = index.ntotal - 1
        metadata[str(vector_id)] = {
            'file_name': file_name,
            'text': translated_text,
        }
        save_faiss_index(index)
        save_metadata(metadata)
    print(f"Stored embedding and metadata for file: {file_name}")

# == AssemblyAI Integration ==

ASSEMBLYAI_UPLOAD_ENDPOINT = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_ENDPOINT = "https://api.assemblyai.com/v2/transcript"
headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}

def upload_to_assemblyai(file_path):
    def read_file(filename, chunk_size=5242880):
        with open(filename, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data
    upload_response = requests.post(
        ASSEMBLYAI_UPLOAD_ENDPOINT,
        headers={"authorization": ASSEMBLYAI_API_KEY},
        data=read_file(file_path)
    )
    upload_response.raise_for_status()
    return upload_response.json()['upload_url']

def request_transcription(upload_url, lang="hi"):
    json_data = {
        "audio_url": upload_url,
        "language_code": lang
    }
    r = requests.post(
        ASSEMBLYAI_TRANSCRIPT_ENDPOINT,
        json=json_data,
        headers=headers
    )
    r.raise_for_status()
    return r.json()['id']

def get_transcription_result(transcript_id):
    polling_endpoint = f"{ASSEMBLYAI_TRANSCRIPT_ENDPOINT}/{transcript_id}"
    while True:
        r = requests.get(polling_endpoint, headers=headers)
        r.raise_for_status()
        data = r.json()
        if data['status'] == 'completed':
            return data['text']
        elif data['status'] == 'error':
            raise RuntimeError(f"Transcription failed: {data['error']}")
        else:
            time.sleep(5)

def transcript_exists_in_drive(txt_filename, transcript_folder_id, retries=3, delay=2):
    service = get_drive_service()
    if not transcript_folder_id:
        return False
    query = f"'{transcript_folder_id}' in parents and mimeType='text/plain' and name='{txt_filename}'"
    for attempt in range(retries):
        results = service.files().list(
            q=query,
            fields='files(id)',
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        files = results.get('files', [])
        if len(files) == 0:
            return False  # Definitely no transcript
        # maybe wait then retry to confirm deletion consistency
        if attempt < retries-1:
            time.sleep(delay)
    return True  # Found transcript after retries

def process_single_video(file_id, file_name, transcript_folder_id, lang="hi", force_translate=True):
    txt_filename = os.path.splitext(file_name)[0] + '.txt'
    if transcript_exists_in_drive(txt_filename, transcript_folder_id):
        print(f"Transcript for '{file_name}' already exists. Skipping transcription.")
        return

    video_path = f"tmp/{file_name}"
    txt_path = None
    try:
        download_video(file_id, video_path)
        upload_url = upload_to_assemblyai(video_path)
        transcript_id = request_transcription(upload_url, lang=lang)
        transcript_text = get_transcription_result(transcript_id)

        transcript_text_en = translate_with_gemini(transcript_text)
        store_transcript_embedding(file_name, transcript_text_en)

        txt_path = f"tmp/{txt_filename}"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text_en)

        upload_transcription(txt_filename, txt_path, transcript_folder_id)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if txt_path and os.path.exists(txt_path):
            os.remove(txt_path)

def transcribe_all_videos(video_folder_id, transcript_folder_id, lang="hi", translate=False):
    videos = list_videos(video_folder_id)
    print(len(videos))
    print(videos)
    for video in videos:
        process_single_video(video['id'], video['name'], transcript_folder_id, lang, force_translate=True)

# == Routes ==

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        raw_video_folder = request.form['video_folder_id']
        raw_transcript_folder = request.form['transcript_folder_id']
        video_folder_id = extract_folder_id(raw_video_folder)
        transcript_folder_id = extract_folder_id(raw_transcript_folder)
        session['video_folder_id'] = video_folder_id
        session['transcript_folder_id'] = transcript_folder_id
        flash("Folder & transcription settings saved successfully!")
        return redirect(url_for('index'))

    videos = []
    if 'video_folder_id' in session and session['video_folder_id']:
        try:
            videos = list_videos(session['video_folder_id'])
            transcribed_files = set()
            if 'transcript_folder_id' in session and session['transcript_folder_id']:
                transcribed_files = list_transcripts(session['transcript_folder_id'])
            for video in videos:
                transcript_name = f"{os.path.splitext(video['name'])[0]}.txt"
                video['transcribed'] = transcript_name in transcribed_files
        except Exception as e:
            flash(f"Drive error: {e}")

    # Store videos in session for use in transcription_status route
    session['videos'] = videos

    return render_template(
        'index.html',
        videos=videos,
        video_folder_id=session.get('video_folder_id', ''),
        transcript_folder_id=session.get('transcript_folder_id', '')
    )

@app.route('/transcribe/<file_id>/<file_name>')
def transcribe(file_id, file_name):
    with processing_videos_lock:
        if file_id in processing_videos:
            flash(f"Transcription already in progress for '{file_name}'.")
            return redirect(url_for('index'))
        processing_videos.add(file_id)

    transcript_folder_id = session.get('transcript_folder_id')
    lang = session.get('language', 'hi')
    translate = session.get('translate', False)

    def run_transcription():
        try:
            process_single_video(file_id, file_name, transcript_folder_id, lang, translate)
        finally:
            with processing_videos_lock:
                processing_videos.discard(file_id)

    threading.Thread(target=run_transcription, daemon=True).start()

    flash(f"Transcription started for '{file_name}'.")
    return redirect(url_for('index'))

@app.route('/transcription_status', methods=['POST'])
def transcription_status():
    video_ids = request.json.get('video_ids', [])
    statuses = {}
    video_folder_id = session.get('video_folder_id')

    if not video_folder_id:
        return jsonify({vid: 'not_transcribed' for vid in video_ids})

    # Fetch fresh videos from Drive
    videos = list_videos(video_folder_id)
    transcript_folder_id = session.get('transcript_folder_id')

    for vid in video_ids:
        with processing_videos_lock:
            if vid in processing_videos:
                statuses[vid] = 'processing'
                continue

        txt_name = None
        for v in videos:
            if v['id'] == vid:
                txt_name = os.path.splitext(v['name'])[0] + '.txt'
                break

        if txt_name and transcript_folder_id:
            exists = transcript_exists_in_drive(txt_name, transcript_folder_id)
            statuses[vid] = 'transcribed' if exists else 'not_transcribed'
        else:
            statuses[vid] = 'not_transcribed'

    return jsonify(statuses)

@app.route('/transcribe_all', methods=['POST'])
def transcribe_all():
    video_folder_id = session.get('video_folder_id')
    transcript_folder_id = session.get('transcript_folder_id')
    lang = session.get('language', 'hi')
    translate = session.get('translate', False)
    threading.Thread(
        target=transcribe_all_videos,
        args=(video_folder_id, transcript_folder_id, lang, translate),
        daemon=True
    ).start()
    flash('Transcription for all videos started.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.getenv("API_KEY"):
        print("WARNING: Missing Gemini API_KEY in environment.")
    if not ASSEMBLYAI_API_KEY:
        print("WARNING: Missing ASSEMBLYAI_API_KEY in environment.")
    app.run(debug=True)
