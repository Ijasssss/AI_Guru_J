# AI Guru J — Backend API (Test Instructions)

Quick steps to run and test the backend locally.

Prerequisites
- Python 3.10+ (use virtualenv / conda)
- ffmpeg (for audio processing if used)
- An audio sample file for testing (e.g. sample.wav)

1) Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

2) Provide environment variables
- Create a `.env` file in the `backend/` folder with placeholders (do NOT commit):
```
OPENAI_API_KEY="your_openai_key_here"
MONGODB_URI="your_mongodb_uri_here"
```
- Ensure `.env` is in `.gitignore`.

3) Start the server
```bash
# from backend/ directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4) Open interactive docs
- Visit: http://127.0.0.1:8000/docs

5) Quick debug check (DB + DNS)
- GET debug status:
```bash
curl http://127.0.0.1:8000/debug/db
```
- Response fields: `initialized`, `reachable`, `host`, `srv_records` / `a_records` or error messages.

6) Test tutor query endpoint (mock pipeline)
```bash
curl -X POST "http://127.0.0.1:8000/api/tutor/query/" \
  -F "audio_file=@/path/to/sample.wav"
```
- Expected JSON keys: `query`, `explanation_text`, `lip_sync_json`, `audio_url`

7) (Optional) Serve / download audio
- If the project exposes `/api/tutor/audio_stream/`, request it:
```bash
curl http://127.0.0.1:8000/api/tutor/audio_stream/ --output response.mp3
```

8) Run project tests / helper scripts
```bash
# if test_api.py is provided and executable
python test_api.py
# or, if you use pytest
pytest
```

Troubleshooting
- "MongoDB client failed to initialize" or DNS timeouts:
  - Ensure `.env` is loaded (export variables) and DNS resolves your Atlas host:
    dig +short <your-cluster-host>
  - Try temporary DNS (macOS):
    sudo networksetup -setdnsservers Wi-Fi 1.1.1.1 8.8.8.8
  - Whitelist your IP in MongoDB Atlas Network Access or use 0.0.0.0/0 for testing.
- Never commit keys. Rotate keys if accidentally exposed.

If you want, I can:
- Add an audio_stream endpoint that saves/serves the last generated MP3, or
- Update README at the repo root instead. Which do you prefer?