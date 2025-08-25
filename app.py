import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import threading
from transcribe import (
    extract_folder_id, list_videos, list_transcripts, process_single_video,
    transcript_exists_in_drive, transcribe_all_videos
)
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='/home/it/Assignment/transcription_service/templates', static_folder='static')
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")

# Keep track of videos being processed to avoid duplicate work
processing_videos = set()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Save folder IDs from form submission
        raw_video_folder = request.form.get('video_folder_id', '')
        raw_transcript_folder = request.form.get('transcript_folder_id', '')
        video_folder_id = extract_folder_id(raw_video_folder)
        transcript_folder_id = extract_folder_id(raw_transcript_folder)
        session['video_folder_id'] = video_folder_id
        session['transcript_folder_id'] = transcript_folder_id
        flash("Folder & transcription settings saved successfully!")
        return redirect(url_for('index'))

    videos = []
    if session.get('video_folder_id'):
        try:
            videos = list_videos(session['video_folder_id'])
            transcribed_files = set()
            if session.get('transcript_folder_id'):
                transcribed_files = list_transcripts(session['transcript_folder_id'])
            for video in videos:
                transcript_name = f"{os.path.splitext(video['name'])[0]}.txt"
                video['transcribed'] = transcript_name in transcribed_files
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
        flash(f"Transcription already in progress for '{file_name}'.")
        return redirect(url_for('index'))

    processing_videos.add(file_id)
    transcript_folder_id = session.get('transcript_folder_id')
    lang = session.get('language', 'hi')
    translate = session.get('translate', False)

    def run_transcription():
        try:
            process_single_video(file_id, file_name, transcript_folder_id, lang, force_translate=translate)
        finally:
            processing_videos.discard(file_id)

    threading.Thread(target=run_transcription, daemon=True).start()

    flash(f"Transcription started for '{file_name}'.")
    return redirect(url_for('index'))


@app.route('/transcription_status', methods=['POST'])
def transcription_status():
    video_ids = request.json.get('video_ids', [])
    statuses = {}
    for vid in video_ids:
        if vid in processing_videos:
            statuses[vid] = 'processing'
        else:
            txt_name = None
            for v in session.get('videos', []):
                if v['id'] == vid:
                    txt_name = os.path.splitext(v['name'])[0] + '.txt'
                    break
            already_transcribed = False
            if txt_name and session.get('transcript_folder_id'):
                already_transcribed = transcript_exists_in_drive(txt_name, session['transcript_folder_id'])
            statuses[vid] = 'transcribed' if already_transcribed else 'not_transcribed'
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
    if not os.getenv("ASSEMBLYAI_API_KEY"):
        print("WARNING: Missing ASSEMBLYAI_API_KEY in environment.")
    app.run(host='0.0.0.0', port=5000, debug=True)
