import os
import re
import time
import threading
import requests
import tempfile
import faiss
import numpy as np
import json
import subprocess
import math
from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.auth import default as google_auth_default
from google.cloud import storage
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# == CONFIGURATION ==
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/devstorage.read_write"
]
creds, project = google_auth_default(scopes=SCOPES)


# Initialize Google Drive client
drive_service = build('drive', 'v3', credentials=creds)

# Cloud Storage Configuration
BUCKET_NAME = os.getenv('STORAGE_BUCKET', 'staff-471204.appspot.com')
VECTOR_INDEX_BLOB = 'shared_data/vector_index.faiss'
METADATA_BLOB = 'shared_data/metadata.json'

# Initialize Cloud Storage client
storage_client = storage.Client(credentials=creds)
bucket = storage_client.bucket(BUCKET_NAME)

# Gemini
genai.configure(api_key=os.getenv("API_KEY"))

# AssemblyAI
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

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

# Decorator for login
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
        "Translate the text (each and every line) given to you to English. "
        "No numbering; output as a single paragraph.\n\n"
        f"Text:\n{text}"
    )
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return (response.text or "").strip()

# == Helper: Extract Folder ID ==
def extract_folder_id(input_str):
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', input_str)
    if match:
        return match.group(1)
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
    names = set()
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{transcript_folder_id}' in parents and mimeType='text/plain'",
            spaces='drive',
            fields='nextPageToken, files(name)',
            pageToken=page_token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True
        ).execute()
        for f in resp.get('files', []):
            names.add(f['name'])
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return names

def download_video(file_id, target_name):
    service = get_drive_service()
    req = service.files().get_media(fileId=file_id)
    temp_dir = '/tmp'
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, target_name)
    with open(path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return path

def upload_transcription(file_name, text_path, transcript_folder_id):
    service = get_drive_service()
    if not transcript_folder_id:
        return
    metadata = {'name': file_name, 'parents': [transcript_folder_id], 'mimeType': 'text/plain'}
    media = MediaFileUpload(text_path, mimetype='text/plain')
    service.files().create(body=metadata, media_body=media, supportsAllDrives=True).execute()

# == FAISS + Cloud Storage ==
def load_faiss_index():
    try:
        blob = bucket.blob(VECTOR_INDEX_BLOB)
        if blob.exists():
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.faiss')
            blob.download_to_filename(tmp.name)
            idx = faiss.read_index(tmp.name)
            os.unlink(tmp.name)
            return idx
    except Exception:
        pass
    return faiss.IndexFlatL2(EMBEDDING_DIM)

def save_faiss_index(index):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.faiss')
    faiss.write_index(index, tmp.name)
    bucket.blob(VECTOR_INDEX_BLOB).upload_from_filename(tmp.name)
    os.unlink(tmp.name)

def load_metadata():
    try:
        blob = bucket.blob(METADATA_BLOB)
        if blob.exists():
            return json.loads(blob.download_as_text())
    except Exception:
        pass
    return {}

def save_metadata(meta):
    bucket.blob(METADATA_BLOB).upload_from_string(json.dumps(meta), content_type='application/json')

def generate_text_embedding(text: str):
    resp = genai.embed_content(model="models/embedding-001", content=text)
    return np.array(resp['embedding'], dtype=np.float32)

def store_transcript_embedding(file_name, text):
    with index_lock:
        idx = load_faiss_index()
        meta = load_metadata()
        emb = generate_text_embedding(text)
        idx.add(np.expand_dims(emb, axis=0))
        vid = idx.ntotal - 1
        meta[str(vid)] = {'file_name': file_name, 'text': text}
        save_faiss_index(idx)
        save_metadata(meta)

# == AssemblyAI Integration ==
ASSEMBLYAI_UPLOAD_ENDPOINT = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_ENDPOINT = "https://api.assemblyai.com/v2/transcript"
headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}

def upload_to_assemblyai(file_path):
    def read_file(fn, chunk_size=5242880):
        with open(fn, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    resp = requests.post(ASSEMBLYAI_UPLOAD_ENDPOINT, headers=headers, data=read_file(file_path))
    resp.raise_for_status()
    return resp.json()['upload_url']

def request_transcription(upload_url, lang="hi"):
    data = {"audio_url": upload_url, "language_code": lang}
    r = requests.post(ASSEMBLYAI_TRANSCRIPT_ENDPOINT, json=data, headers=headers)
    r.raise_for_status()
    return r.json()['id']

def get_transcription_result(transcript_id):
    url = f"{ASSEMBLYAI_TRANSCRIPT_ENDPOINT}/{transcript_id}"
    while True:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        d = r.json()
        if d['status'] == 'completed':
            return d['text']
        if d['status'] == 'error':
            raise RuntimeError(d['error'])
        time.sleep(5)

def transcript_exists_in_drive(txt_name, folder_id, retries=3, delay=2):
    service = get_drive_service()
    query = f"'{folder_id}' in parents and mimeType='text/plain' and name='{txt_name}'"
    for i in range(retries):
        res = service.files().list(q=query, fields='files(id)', includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
        if not res.get('files'):
            return False
        time.sleep(delay)
    return True

# == Video Processing with Chunking ==
def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", # use 'error' to suppress info logs
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"[WARN] ffprobe failed for {video_path}: {e}")
        return 0.0

def split_video_into_chunks(video_path, chunk_duration_minutes=5):
    """Split video into smaller chunks; if duration is invalid, return the original video."""
    duration = get_video_duration(video_path)
    chunk_secs = chunk_duration_minutes * 60

    if duration <= 0 or duration <= chunk_secs:
        print(f"[INFO] Video duration ({duration}s) is short. No chunking needed.")
        return [video_path]

    num_chunks = math.ceil(duration / chunk_secs)
    chunks = []
    base = os.path.splitext(video_path)[0]

    for i in range(num_chunks):
        start = i * chunk_secs
        chunk_path = f"{base}_chunk_{i+1:03d}.mp4"

        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start),
            "-t", str(chunk_secs),
            "-i", video_path, "-c", "copy", chunk_path
        ], check=True)

        print(f"[INFO] Chunk {i+1} created: {chunk_path}")
        chunks.append(chunk_path)
    return chunks

def process_single_video_chunked(file_id, file_name, transcript_folder_id, lang="hi"):
    """Process video in chunks to avoid timeout"""
    
    # Check if transcript already exists
    txt_name = os.path.splitext(file_name)[0] + '.txt'
    if transcript_exists_in_drive(txt_name, transcript_folder_id):
        print(f"[INFO] Transcript for '{file_name}' already exists. Skipping transcription.")
        return
    
    chunks = []
    video_path = None
    
    try:
        # Download original video
        print(f"[DEBUG] Downloading video: {file_name}")
        video_path = download_video(file_id, file_name)
        
        # Split into 5-minute chunks
        print(f"[DEBUG] Splitting video into chunks")
        chunks = split_video_into_chunks(video_path, chunk_duration_minutes=5)
        print(f"[DEBUG] Created {len(chunks)} chunks")
        
        all_transcripts = []
        
        # Process each chunk
        for i, chunk_path in enumerate(chunks):
            print(f"[DEBUG] Processing chunk {i+1}/{len(chunks)}")
            
            # Convert chunk to audio
            audio_path = chunk_path.replace('.mp4', '.wav')
            subprocess.run([
                "ffmpeg", "-y", "-i", chunk_path, "-ac", "1", "-ar", "16000", audio_path
            ], check=True)
            
            # Transcribe chunk
            upload_url = upload_to_assemblyai(audio_path)
            transcript_id = request_transcription(upload_url, lang=lang)
            chunk_text = get_transcription_result(transcript_id)
            all_transcripts.append(chunk_text)
            
            print(f"[DEBUG] Chunk {i+1} transcribed: {len(chunk_text)} characters")
            
            # Clean up chunk files immediately to save space
            os.remove(chunk_path)
            os.remove(audio_path)
        
        # Combine all transcripts
        combined_transcript = " ".join(all_transcripts)
        print(f"[DEBUG] Combined transcript length: {len(combined_transcript)} characters")
        
        # Translate combined transcript
        print(f"[DEBUG] Translating transcript with Gemini")
        translated_text = translate_with_gemini(combined_transcript)
        
        # Store embedding
        print(f"[DEBUG] Storing embedding and metadata")
        store_transcript_embedding(file_name, translated_text)
        
        # Save and upload final transcript
        txt_path = f"/tmp/{txt_name}"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(translated_text)
        upload_transcription(txt_name, txt_path, transcript_folder_id)
        
        print(f"[SUCCESS] Transcript for {file_name} processed and uploaded.")
        
        # Clean up
        os.remove(txt_path)
        
    except Exception as e:
        print(f"[ERROR] Error processing {file_name}: {e}")
        raise e
    finally:
        # Clean up on error or completion
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        for chunk_path in chunks:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            audio_path = chunk_path.replace('.mp4', '.wav')
            if os.path.exists(audio_path):
                os.remove(audio_path)

# Keep the original function for backward compatibility or smaller videos
def process_single_video(file_id, file_name, transcript_folder_id, lang="hi"):
    """Process single video without chunking - for smaller videos or testing"""
    # Check if transcript already exists
    txt_name = os.path.splitext(file_name)[0] + '.txt'
    if transcript_exists_in_drive(txt_name, transcript_folder_id):
        print(f"[INFO] Transcript for '{file_name}' already exists. Skipping transcription.")
        return
    
    # Download video from Google Drive to /tmp
    video_path = download_video(file_id, file_name)
    # Define output audio path in /tmp with .wav extension
    audio_path = f"/tmp/{os.path.splitext(file_name)[0]}.wav"

    # Convert MP4 → WAV using subprocess to avoid shell quoting issues
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-ac", "1",
            "-ar", "16000",
            audio_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e}")

    # Continue with AssemblyAI upload, transcription, translation, etc.
    upload_url = upload_to_assemblyai(audio_path)
    transcript_id = request_transcription(upload_url, lang=lang)
    transcript_text = get_transcription_result(transcript_id)
    translated_text = translate_with_gemini(transcript_text)
    store_transcript_embedding(file_name, translated_text)

    # Save and upload transcript text
    txt_path = f"/tmp/{os.path.splitext(file_name)[0]}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(translated_text)
    upload_transcription(os.path.basename(txt_path), txt_path, transcript_folder_id)

    # Clean up temp files
    for path in (video_path, audio_path, txt_path):
        if os.path.exists(path):
            os.remove(path)

def transcribe_all_videos(vf, tf, lang="hi"):
    for v in list_videos(vf):
        # Use chunked processing for all videos to avoid timeouts
        process_single_video_chunked(v['id'], v['name'], tf, lang)

# == Flask Routes ==

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form.get('username')
        p = request.form.get('password')
        if u == VALID_USERNAME and p == VALID_PASSWORD:
            session['logged_in'] = True
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        return render_template('index.html', error='Invalid credentials.', session={'logged_in': False})
    if session.get('logged_in'):
        return redirect(url_for('index'))
    return render_template('index.html', session={'logged_in': False})

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        vf = extract_folder_id(request.form['video_folder_id'])
        tf = extract_folder_id(request.form['transcript_folder_id'])
        session['video_folder_id'] = vf
        session['transcript_folder_id'] = tf
        flash('Settings saved.')
        return redirect(url_for('index'))
    vids = []
    vf = session.get('video_folder_id')
    tf = session.get('transcript_folder_id')
    if vf:
        vids = list_videos(vf)
        done = list_transcripts(tf) if tf else set()
        for v in vids:
            v['transcribed'] = f"{os.path.splitext(v['name'])[0]}.txt" in done
    session['videos'] = vids
    return render_template('index.html', videos=vids, video_folder_id=vf or '', transcript_folder_id=tf or '', session=session)

@app.route('/transcribe/<file_id>/<file_name>')
@login_required
def transcribe(file_id, file_name):
    from urllib.parse import unquote
    fn = unquote(file_name)
    with processing_videos_lock:
        if file_id in processing_videos:
            flash(f"Already processing '{fn}'.")
            return redirect(url_for('index'))
        processing_videos.add(file_id)
    try:
        # Use chunked processing to avoid timeouts
        process_single_video_chunked(file_id, fn, session.get('transcript_folder_id'), session.get('language', 'hi'))
        flash(f"Completed '{fn}'.")
    except Exception as e:
        flash(f"Error: {e}")
    finally:
        processing_videos.discard(file_id)
    return redirect(url_for('index'))

@app.route('/transcription_status', methods=['POST'])
@login_required
def transcription_status():
    ids = request.json.get('video_ids', [])
    statuses = {}
    vf = session.get('video_folder_id')
    tf = session.get('transcript_folder_id')
    vids = list_videos(vf) if vf else []
    for vid in ids:
        if vid in processing_videos:
            statuses[vid] = 'processing'
        else:
            name = next((v['name'] for v in vids if v['id'] == vid), None)
            txt = f"{os.path.splitext(name)[0]}.txt" if name else None
            statuses[vid] = 'transcribed' if txt and transcript_exists_in_drive(txt, tf) else 'not_transcribed'
    return jsonify(statuses)

@app.route('/transcribe_all', methods=['POST'])
@login_required
def transcribe_all():
    transcribe_all_videos(session.get('video_folder_id'), session.get('transcript_folder_id'), session.get('language', 'hi'))
    flash('Batch transcription started.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))