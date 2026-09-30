from pathlib import Path
import logging
import os
import re

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging


try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ============================================================
# CONFIGURAZIONE
# ============================================================

REF_AUDIO_PATH = "cutdonna.wav"
REF_TEXT_PATH = "testodonna.txt"
INPUT_TEXT_PATH = "testo_da_leggere.txt"

# Audio originale (testo così com'è) + audio ottimizzato via GPT
OUTPUT_ORIGINAL_PATH = "ELIMINA1.wav"
OUTPUT_OPTIMIZED_PATH = "EL2.wav"
OPTIMIZED_TEXT_PATH = "EL3.txt"

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
GPT_REWRITE_MODEL = "gpt-3.5-turbo"

MAX_CHARS_PER_CHUNK = 120
PAUSE_MS_BETWEEN_CHUNKS = 250


# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================

def normalize_text(text: str) -> str:
    text = text.strip()
    replacements = {
        r"\bAI\b": "intelligenza artificiale",
        r"\bLLM\b": "elle elle emme",
        r"\bAPI\b": "a pi i",
        r"\becc\.\b": "eccetera",
        r"\bsig\.\b": "signore",
        r"\bsig\.ra\b": "signora",
        r"\bdott\.\b": "dottore",
        r"\bdr\.\b": "dottore",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", text)
    return text.strip()


def split_text_into_chunks(text: str, max_chars: int = 120) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        proposed = f"{current_chunk} {sentence}".strip()
        if len(proposed) <= max_chars:
            current_chunk = proposed
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def create_silence(ms: int, sample_rate: int) -> np.ndarray:
    num_samples = int(sample_rate * ms / 1000)
    return np.zeros(num_samples, dtype=np.float32)


def normalize_audio(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio)))
    if max_abs > 0:
        audio = (audio / max_abs) * peak
    return audio.astype(np.float32)


def build_qwen_tts_prompt(text: str) -> list[dict]:
    """
    Prompt per ottenere una versione più naturale in sintesi vocale.
    Mantiene significato e registro, ma ottimizza ritmo, pause e fluidità.
    """
    system_prompt = (
        "Sei un editor professionista per text-to-speech in italiano. "
        "Riscrivi il testo per ottenere una voce più umana, calda e naturale su Qwen TTS, "
        "senza cambiare il significato. "
        "Obiettivi: migliorare prosodia, ritmo, pause respiratorie e scorrevolezza. "
        "Regole operative: "
        "1) Mantieni contenuto, tono e informazioni originali. "
        "2) Spezza periodi troppo lunghi in frasi brevi o medie. "
        "3) Inserisci punteggiatura utile alla voce: "
        "virgole per micro-pause, punto per pause nette, due punti per introduzioni, "
        "punto e virgola solo se davvero utile. "
        "4) Evita accumuli di subordinate e costruzioni contorte. "
        "5) Trasforma simboli, abbreviazioni e sigle in forme leggibili ad alta voce. "
        "6) Mantieni uno stile parlato curato, non teatrale e non robotico. "
        "7) Evita parentesi, virgolette inutili, elenchi puntati e markup. "
        "8) Non aggiungere informazioni nuove e non rimuovere concetti importanti. "
        "Output: restituisci solo il testo finale ottimizzato, in italiano."
    )
    user_prompt = (
        "Ottimizza questo testo per la lettura vocale naturale.\n\n"
        f"{text}\n\n"
        "Ricontrolla prima di rispondere: il testo deve essere fluido all'ascolto."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def rewrite_text_for_qwen_tts(text: str) -> str:
    if OpenAI is None:
        raise RuntimeError(
            "Pacchetto 'openai' non installato. Installa con: pip install openai"
        )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY non trovato. Imposta la variabile d'ambiente prima di eseguire."
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=GPT_REWRITE_MODEL,
        messages=build_qwen_tts_prompt(text),
        temperature=0.2,
    )

    optimized = response.choices[0].message.content if response.choices else ""
    optimized = (optimized or "").strip()
    if not optimized:
        raise RuntimeError("GPT non ha restituito testo valido per l'ottimizzazione.")
    return optimized


def load_qwen_model():
    print("Caricamento modello Qwen TTS...")
    transformers_logging.set_verbosity_error()
    logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
    model = load_model(MODEL_ID)

    # Fix del tokenizer per evitare il warning regex Mistral e tokenizzazione errata.
    try:
        tokenizer_source = getattr(getattr(model, "config", None), "model_path", MODEL_ID)
        model.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            fix_mistral_regex=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"Attenzione: tokenizer non ricaricato con fix regex: {exc}")

    transformers_logging.set_verbosity_warning()
    print("Modello caricato.")
    return model


def synthesize_to_wav(
    *,
    model,
    ref_audio_path: str,
    ref_text: str,
    text: str,
    output_path: str,
):
    clean_text = normalize_text(text)
    if not clean_text:
        raise RuntimeError("Testo vuoto dopo normalizzazione.")

    text_chunks = split_text_into_chunks(clean_text, max_chars=MAX_CHARS_PER_CHUNK)
    if not text_chunks:
        raise RuntimeError("Nessun blocco testuale valido per la generazione.")

    all_audio_chunks = []
    sample_rate = None

    print(f"\nGenerazione '{output_path}' - blocchi: {len(text_chunks)}")
    for idx, text_chunk in enumerate(text_chunks, start=1):
        print(f"Blocco {idx}/{len(text_chunks)}: {text_chunk}")
        results = list(
            model.generate(
                text=text_chunk,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
            )
        )

        if not results:
            continue

        local_audio_chunks = []
        for result in results:
            chunk = np.asarray(result.audio)
            if chunk.ndim > 1:
                chunk = np.squeeze(chunk)
            if chunk.size > 0:
                local_audio_chunks.append(chunk.astype(np.float32))
            if sample_rate is None:
                sample_rate = getattr(result, "sample_rate", None)

        if not local_audio_chunks:
            continue

        block_audio = np.concatenate(local_audio_chunks).astype(np.float32)
        all_audio_chunks.append(block_audio)

        if idx < len(text_chunks):
            if sample_rate is None:
                sample_rate = getattr(model, "sample_rate", 24000)
            all_audio_chunks.append(create_silence(PAUSE_MS_BETWEEN_CHUNKS, sample_rate))

    if not all_audio_chunks:
        raise RuntimeError(f"Il modello non ha prodotto audio valido per '{output_path}'.")
    if sample_rate is None:
        sample_rate = getattr(model, "sample_rate", 24000)

    final_audio = np.concatenate(all_audio_chunks)
    final_audio = normalize_audio(final_audio, peak=0.95)
    sf.write(output_path, final_audio, sample_rate, subtype="PCM_16")
    duration_sec = len(final_audio) / sample_rate
    print(f"Salvato: {output_path} ({duration_sec:.2f}s)")


# ============================================================
# MAIN
# ============================================================

ref_text = Path(REF_TEXT_PATH).read_text(encoding="utf-8").strip()
if not ref_text:
    raise RuntimeError(f"Il file {REF_TEXT_PATH} è vuoto.")
ref_text = normalize_text(ref_text)

source_text = Path(INPUT_TEXT_PATH).read_text(encoding="utf-8").strip()
if not source_text:
    raise RuntimeError(f"Il file {INPUT_TEXT_PATH} è vuoto.")

print("Ottimizzazione testo via GPT...")
optimized_text = rewrite_text_for_qwen_tts(source_text)
Path(OPTIMIZED_TEXT_PATH).write_text(optimized_text, encoding="utf-8")
print(f"Testo ottimizzato salvato in: {OPTIMIZED_TEXT_PATH}")

model = load_qwen_model()

# Audio 1: testo originale
synthesize_to_wav(
    model=model,
    ref_audio_path=REF_AUDIO_PATH,
    ref_text=ref_text,
    text=source_text,
    output_path=OUTPUT_ORIGINAL_PATH,
)

# Audio 2 (in più): testo ottimizzato via GPT
synthesize_to_wav(
    model=model,
    ref_audio_path=REF_AUDIO_PATH,
    ref_text=ref_text,
    text=optimized_text,
    output_path=OUTPUT_OPTIMIZED_PATH,
)

print("\nCompletato.")
print(f"Audio originale: {OUTPUT_ORIGINAL_PATH}")
print(f"Audio ottimizzato GPT: {OUTPUT_OPTIMIZED_PATH}")
print(f"Testo ottimizzato: {OPTIMIZED_TEXT_PATH}")
