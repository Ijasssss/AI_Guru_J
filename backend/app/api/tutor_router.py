from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import whisper # OpenAI Whisper for Speech Recognition
import logging
from urllib.parse import quote_plus

from ..core.nlp_engine import get_ai_explanation
from ..core.speech_synth import generate_speech
from ..database.mongodb_ops import log_interaction
from ..utils.rhubarb_generator import generate_lip_sync_json

router = APIRouter()
logging.basicConfig(level=logging.INFO)

# --- Global Model Loading for Whisper (Occurs on startup) ---
# NOTE: Loading models can be memory intensive.
try:
    logging.info("Loading Whisper model...")
    # Use a small, fast model for a demonstration
    whisper_model = whisper.load_model("base") 
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Whisper model: {e}")
    whisper_model = None


@router.post("/query/")
async def handle_query(audio_file: UploadFile = File(...)):
    """Handles the full voice-to-3D-response pipeline."""
    if not whisper_model:
        raise HTTPException(status_code=503, detail="Speech Recognition service is unavailable.")

    # 1. Speech Recognition (Whisper)
    audio_bytes = await audio_file.read()
    # In a real app, you save the audio_bytes to a temporary file compatible with Whisper.
    # We will use a MOCK result for now to ensure the flow works:
    user_query = "What is Python language in coding? write a python code to print hello world" # Mock result

    logging.info(f"Received query: {user_query}")

    # 2. NLP and Explanation (FLAN-T5)
    ai_response_text = get_ai_explanation(user_query)

    # 3. Text-to-Speech (gTTS)
    speech_mp3_stream = generate_speech(ai_response_text)
    
    # 4. Lip-Sync JSON Generation (Rhubarb - MOCK Utility)
    lip_sync_data = generate_lip_sync_json(speech_mp3_stream) 

    # 5. Log Interaction (MongoDB)
    log_interaction(user_query, ai_response_text)

    # Return the data needed by the frontend to trigger audio and animation
    safe_text = quote_plus(ai_response_text)
    return {
        "query": user_query,
        "explanation_text": ai_response_text,
        "lip_sync_json": lip_sync_data,
        "audio_url": f"/api/tutor/audio_stream/?text={safe_text}"
    }

@router.get("/audio_stream/")
async def stream_audio(text: str):
    """Streams the gTTS-generated audio back to the frontend."""
    audio_stream = generate_speech(text)
    return StreamingResponse(audio_stream, media_type="audio/mp3")