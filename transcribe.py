import os
import re
import time
import threading
import requests
import tempfile
import faiss
import numpy as np
import json
from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account
from google.cloud import storage
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# == CONFIGURATION ==
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '/home/it/staff_ass/transcribe/service_account.json')
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Cloud Storage Configuration
BUCKET_NAME = os.getenv('STORAGE_BUCKET', 'staff-471204.appspot.com')
VECTOR_INDEX_BLOB = 'shared_data/vector_index.faiss'
METADATA_BLOB = 'shared_data/metadata.json'

# Initialize Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# Gemini
genai.configure(api_key=os.getenv("API_KEY"))

# Flask
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")

# GLOBAL VARIABLES
EMBEDDING_DIM = 768  # For most Gemini embedding models
index_lock = threading.Lock()
processing_videos = set()
processing_videos_lock = threading.Lock()

# Credentials for login
VALID_USERNAME = 'GeniusLane'
VALID_PASSWORD = 'Geniuslane'

# DEFINE THE DECORATOR FIRST - before using it
def login_required(func):
    from functools import wraps
    @wraps(func)
    def decorated_view(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    return decorated_view

# == Google Drive Service ==
def get_drive_service():
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
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', input_str)
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
    
    # Use /tmp directory which is writable in App Engine
    temp_dir = '/tmp'
    os.makedirs(temp_dir, exist_ok=True)
    full_path = os.path.join(temp_dir, os.path.basename(target_path))
    
    with open(full_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return full_path

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

# == CLOUD STORAGE FAISS FUNCTIONS ==
def load_faiss_index():
    try:
        # Download from Cloud Storage to /tmp
        index_blob = bucket.blob(VECTOR_INDEX_BLOB)
        if index_blob.exists():
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.faiss')
            index_blob.download_to_filename(temp_file.name)
            index = faiss.read_index(temp_file.name)
            os.unlink(temp_file.name)  # Clean up
            print(f"FAISS index loaded from Cloud Storage: {VECTOR_INDEX_BLOB}")
            return index
        else:
            print("No existing FAISS index found, creating new one")
            return faiss.IndexFlatL2(EMBEDDING_DIM)
    except Exception as e:
        print(f"Error loading FAISS index from Cloud Storage: {e}")
        return faiss.IndexFlatL2(EMBEDDING_DIM)

def save_faiss_index(index):
    try:
        # Save to /tmp first, then upload to Cloud Storage
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.faiss')
        faiss.write_index(index, temp_file.name)
        
        # Upload to Cloud Storage
        index_blob = bucket.blob(VECTOR_INDEX_BLOB)
        index_blob.upload_from_filename(temp_file.name)
        
        os.unlink(temp_file.name)  # Clean up
        print(f"FAISS index saved successfully to Cloud Storage: {VECTOR_INDEX_BLOB}")
    except Exception as e:
        print(f"Error saving FAISS index to Cloud Storage: {e}")

def load_metadata():
    try:
        metadata_blob = bucket.blob(METADATA_BLOB)
        if metadata_blob.exists():
            content = metadata_blob.download_as_text()
            print(f"Metadata loaded from Cloud Storage: {METADATA_BLOB}")
            return json.loads(content)
        print("No existing metadata found, creating new")
        return {}
    except Exception as e:
        print(f"Error loading metadata from Cloud Storage: {e}")
        return {}

def save_metadata(metadata):
    try:
        metadata_blob = bucket.blob(METADATA_BLOB)
        metadata_blob.upload_from_string(json.dumps(metadata), content_type='application/json')
        print(f"Metadata saved successfully to Cloud Storage: {METADATA_BLOB}")
    except Exception as e:
        print(f"Error saving metadata to Cloud Storage: {e}")

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
        print(f"[DEBUG] FAISS index now contains {index.ntotal} vectors.")
    print(f"[DEBUG] Stored embedding and metadata for file: {file_name}")

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
        if attempt < retries-1:
            time.sleep(delay)
    return True  # Found transcript after retries

def process_single_video(file_id, file_name, transcript_folder_id, lang="hi", force_translate=True):
    print(f"[DEBUG] Starting transcription for video: {file_name}, ID: {file_id}")
    txt_filename = os.path.splitext(file_name)[0] + '.txt'
    
    if transcript_exists_in_drive(txt_filename, transcript_folder_id):
        print(f"[INFO] Transcript for '{file_name}' already exists. Skipping transcription.")
        return
    video_path = f"/tmp/{file_name}"
    txt_path = f"/tmp/{txt_filename}"
    
    try:
        print(f"[DEBUG] Downloading video to {video_path}")
        download_video(file_id, video_path)
        
        print(f"[DEBUG] Uploading video to AssemblyAI")
        upload_url = upload_to_assemblyai(video_path)
        print(f"[DEBUG] AssemblyAI upload URL: {upload_url}")
        
        transcript_id = request_transcription(upload_url, lang=lang)
        print(f"[DEBUG] AssemblyAI transcript ID: {transcript_id}")
        
        transcript_text = get_transcription_result(transcript_id)
        print(f"[DEBUG] Raw transcript received (len={len(transcript_text)}): {transcript_text[:100]}...")
        
        print(f"[DEBUG] Translating transcript with Gemini")
        transcript_text_en = translate_with_gemini(transcript_text)
        print(f"[DEBUG] Translation result (len={len(transcript_text_en)}): {transcript_text_en[:100]}...")
        
        print(f"[DEBUG] Storing embedding and metadata")
        store_transcript_embedding(file_name, transcript_text_en)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text_en)
        
        print(f"[DEBUG] Uploading transcript text file to Google Drive")
        upload_transcription(txt_filename, txt_path, transcript_folder_id)
        
        print(f"[SUCCESS] Transcript for {file_name} processed and uploaded.")
        
    except Exception as e:
        print(f"[ERROR] Error processing {file_name}: {e}")
        raise e
    finally:
        for temp_file in [video_path, txt_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def transcribe_all_videos(video_folder_id, transcript_folder_id, lang="hi", translate=False):
    videos = list_videos(video_folder_id)
    print(f"Found {len(videos)} videos to process")
    for video in videos:
        try:
            process_single_video(video['id'], video['name'], transcript_folder_id, lang, force_translate=True)
        except Exception as e:
            print(f"Error processing video {video['name']}: {e}")
            continue

# == LOGIN ROUTES ==
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session['logged_in'] = True
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            error = 'Invalid credentials. Please try again.'
            return render_template('index.html', error=error, session={'logged_in': False})
    else:
        if session.get('logged_in'):
            return redirect(url_for('index'))
        return render_template('index.html', session={'logged_in': False})

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

# == MAIN ROUTES ==
@app.route('/', methods=['GET', 'POST'])
@login_required
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
    session['videos'] = videos
    return render_template(
        'index.html',
        videos=videos,
        video_folder_id=session.get('video_folder_id', ''),
        transcript_folder_id=session.get('transcript_folder_id', ''),
        session=session
    )

@app.route('/transcribe/<file_id>/<file_name>')
@login_required
def transcribe(file_id, file_name):
    from urllib.parse import unquote
    file_name = unquote(file_name)
    with processing_videos_lock:
        if file_id in processing_videos:
            flash(f"Transcription already in progress for '{file_name}'.")
            return redirect(url_for('index'))
        processing_videos.add(file_id)
    try:
        process_single_video(
            file_id,
            file_name,
            session.get('transcript_folder_id'),
            session.get('language', 'hi'),
            force_translate=True
        )
        flash(f"Transcription completed for '{file_name}'!")
    except Exception as e:
        flash(f"Error transcribing '{file_name}': {str(e)}")
    finally:
        with processing_videos_lock:
            processing_videos.discard(file_id)
    return redirect(url_for('index'))

@app.route('/transcription_status', methods=['POST'])
@login_required
def transcription_status():
    video_ids = request.json.get('video_ids', [])
    statuses = {}
    video_folder_id = session.get('video_folder_id')
    if not video_folder_id:
        return jsonify({vid: 'not_transcribed' for vid in video_ids})
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
@login_required
def transcribe_all():
    video_folder_id = session.get('video_folder_id')
    transcript_folder_id = session.get('transcript_folder_id')
    lang = session.get('language', 'hi')
    translate = session.get('translate', False)
    try:
        transcribe_all_videos(video_folder_id, transcript_folder_id, lang, translate)
        flash('Transcription for all videos completed.')
    except Exception as e:
        flash(f'Error in batch transcription: {str(e)}')
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.getenv("API_KEY"):
        print("WARNING: Missing Gemini API_KEY in environment.")
    if not ASSEMBLYAI_API_KEY:
        print("WARNING: Missing ASSEMBLYAI_API_KEY in environment.")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
