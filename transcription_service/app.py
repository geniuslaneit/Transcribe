import os
from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, flash, session
)
import threading
from transcribe import (
    extract_folder_id, list_videos,
    list_transcripts, process_single_video,
    transcript_exists_in_drive, transcribe_all_videos
)
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2 import service_account

load_dotenv()

# Compute absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
SERVICE_ACCOUNT_FILE = os.getenv('DRIVE_SERVICE_ACCOUNT_JSON')

# Flask app with explicit template/static folders
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")

# Drive API configuration
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build('drive', 'v3', credentials=creds)

def list_videos_in_folder(folder_id):
    drive = get_drive_service()
    resp = drive.files().list(
        q=f"'{folder_id}' in parents and mimeType contains 'video/'",
        fields="files(id,name,webViewLink)"
    ).execute()
    videos = resp.get('files', [])
    print("Drive returned videos:", videos)  # debug log
    return videos

# Track processing to avoid duplicates
processing_videos = set()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        raw_vid_id = request.form.get('video_folder_id', '')
        raw_tx_id  = request.form.get('transcript_folder_id', '')
        vid_id = extract_folder_id(raw_vid_id)
        tx_id  = extract_folder_id(raw_tx_id)
        session['video_folder_id']      = vid_id
        session['transcript_folder_id'] = tx_id
        flash("Folder settings saved.")
        return redirect(url_for('index'))

    videos = []
    if session.get('video_folder_id'):
        folder_id = session['video_folder_id']
        print("Video folder ID in session:", folder_id)  # debug log
        try:
            videos = list_videos_in_folder(folder_id)
            # Mark which videos are already transcribed
            transcribed = set()
            if session.get('transcript_folder_id'):
                transcribed = set(list_transcripts(session['transcript_folder_id']))
            for v in videos:
                name_root = os.path.splitext(v['name'])[0] + '.txt'
                v['transcribed'] = (name_root in transcribed)
        except Exception as e:
            flash(f"Drive error: {e}")

    return render_template(
        'index.html',
        videos=videos,
        video_folder_id=session.get('video_folder_id', ''),
        transcript_folder_id=session.get('transcript_folder_id', '')
    )

@app.route('/transcribe/<file_id>/<file_name>')
def transcribe(file_id, file_name):
    if file_id in processing_videos:
        flash(f"Already processing {file_name}.")
        return redirect(url_for('index'))

    processing_videos.add(file_id)
    tx_folder = session.get('transcript_folder_id')
    lang       = session.get('language', 'hi')
    translate  = session.get('translate', False)

    def run_transcription():
        try:
            process_single_video(
                file_id, file_name, tx_folder, lang,
                force_translate=translate
            )
        finally:
            processing_videos.discard(file_id)

    threading.Thread(target=run_transcription, daemon=True).start()
    flash(f"Started transcription for {file_name}.")
    return redirect(url_for('index'))

@app.route('/transcription_status', methods=['POST'])
def transcription_status():
    video_ids = request.json.get('video_ids', [])
    statuses = {}
    for vid in video_ids:
        if vid in processing_videos:
            statuses[vid] = 'processing'
        else:
            txt = f"{vid}.txt"
            done = False
            if session.get('transcript_folder_id'):
                done = transcript_exists_in_drive(
                    os.path.splitext(txt)[0] + '.txt',
                    session['transcript_folder_id']
                )
            statuses[vid] = 'transcribed' if done else 'not_transcribed'
    return jsonify(statuses)

@app.route('/transcribe_all', methods=['POST'])
def transcribe_all():
    vf = session.get('video_folder_id')
    tf = session.get('transcript_folder_id')
    lang      = session.get('language', 'hi')
    translate = session.get('translate', False)
    threading.Thread(
        target=transcribe_all_videos,
        args=(vf, tf, lang, translate),
        daemon=True
    ).start()
    flash('Transcription for all videos started.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.getenv("API_KEY"):
        print("WARNING: Missing GEMINI API_KEY.")
    if not os.getenv("ASSEMBLYAI_API_KEY"):
        print("WARNING: Missing ASSEMBLYAI_API_KEY.")
    app.run(host='0.0.0.0', port=5000, debug=True)
