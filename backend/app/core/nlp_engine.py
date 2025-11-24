from transformers import T5ForConditionalGeneration, T5TokenizerFast
from .config import settings
import logging
import torch

logging.basicConfig(level=logging.INFO)

# --- Global Model Loading (Occurs on startup) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    logging.info("Loading FLAN-T5 model...")
    tokenizer = T5TokenizerFast.from_pretrained(settings.NLP_MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(settings.NLP_MODEL_ID).to(device)
    model.eval()
    logging.info(f"FLAN-T5 model loaded successfully on {device}.")
except Exception as e:
    logging.error(f"Error loading FLAN-T5: {e}")
    tokenizer = None
    model = None

def get_ai_explanation(user_query: str) -> str:
    """Generates a concise, high-quality explanation and a small Python example."""
    if model is None or tokenizer is None:
        return "Sorry, the AI model is not ready yet."

    prompt = (
        "You are a friendly tutor. Provide a concise explanation (<=200 words) of the concept below, "
        "followed by a short Python code block that demonstrates the idea. "
        "Return plain text with the code enclosed in triple backticks.\n\n"
        f"Concept: {user_query}\n\nExplanation and example:"
    )

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    # Use deterministic beam search for reliable, high-quality outputs
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            min_length=80,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.0
        )

    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return explanation.strip()