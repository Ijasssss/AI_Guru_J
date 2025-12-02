import edge_tts
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

VOICES = {
    "male": "en-IN-PrabhatNeural",
    "female": "en-IN-NeerjaNeural"
}


async def generate_speech(text_to_speak: str, gender: str = "female") -> str:
    """
    Generates speech using Microsoft Edge TTS.
    Returns the PATH to a temporary .mp3 file.
    """
    try:
        if not text_to_speak:
            raise ValueError("No text provided for TTS")

        # 1. Select voice based on gender
        voice = VOICES.get(gender, VOICES["female"])
        
        # 2. Create a temp file to save the audio
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd) 

        # 3. Generate Audio
        communicate = edge_tts.Communicate(text_to_speak, voice)
        await communicate.save(path)

        return path

    except Exception as e:
        logger.error(f"❌ Edge-TTS Error: {e}")
        raise e