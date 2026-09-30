from pathlib import Path
import re

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model


# ============================================================
# CONFIGURAZIONE
# ============================================================

# File audio di riferimento: contiene la voce da imitare/adattare
REF_AUDIO_PATH = "cutdonna.wav"

# Trascrizione esatta dell'audio di riferimento
REF_TEXT_PATH = "testodonna.txt"

# File WAV finale generato
OUTPUT_PATH = "ELIMINA.wav"

# Testo da leggere
TEXT_TO_READ = (
    "Benvenuti al Museo della Sindone di Torino. "
    "Oggi scopriremo insieme la storia della Sindone e del museo. "
    "Io sarò la vostra guida durante questa visita. "
    "Siete pronti? Iniziamo."
)

# Modello TTS MLX
MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

# Lunghezza massima di ogni blocco testuale
MAX_CHARS_PER_CHUNK = 120

# Pausa tra un blocco e il successivo, in millisecondi
PAUSE_MS_BETWEEN_CHUNKS = 250


# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================

def normalize_text(text: str) -> str:
    """
    Rende il testo più facile da leggere per il TTS.
    - Pulisce spazi inutili
    - Espande alcune sigle comuni
    - Sistema la punteggiatura
    """
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

    # Rimuove spazi multipli
    text = re.sub(r"\s+", " ", text)

    # Rimuove spazi prima della punteggiatura
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Aggiunge spazio dopo la punteggiatura se manca
    text = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", text)

    return text.strip()


def split_text_into_chunks(text: str, max_chars: int = 120) -> list[str]:
    """
    Divide il testo in blocchi corti, cercando di rispettare la fine frase.
    Questo aiuta il TTS a leggere in modo più naturale.
    """
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
    """
    Genera un piccolo silenzio tra i blocchi audio.
    """
    num_samples = int(sample_rate * ms / 1000)
    return np.zeros(num_samples, dtype=np.float32)


def normalize_audio(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """
    Normalizza l'audio per evitare clipping e mantenere un buon volume.
    """
    max_abs = float(np.max(np.abs(audio)))
    if max_abs > 0:
        audio = (audio / max_abs) * peak
    return audio.astype(np.float32)


# ============================================================
# LETTURA TESTO DI RIFERIMENTO
# ============================================================

ref_text = Path(REF_TEXT_PATH).read_text(encoding="utf-8").strip()

if not ref_text:
    raise RuntimeError(f"Il file di testo {REF_TEXT_PATH} è vuoto.")

ref_text = normalize_text(ref_text)


# ============================================================
# PREPARAZIONE TESTO DA LEGGERE
# ============================================================

clean_text = normalize_text(TEXT_TO_READ)

if not clean_text:
    raise RuntimeError("Il testo da leggere è vuoto dopo la normalizzazione.")

text_chunks = split_text_into_chunks(clean_text, max_chars=MAX_CHARS_PER_CHUNK)

if not text_chunks:
    raise RuntimeError("Non sono stati creati blocchi testuali validi.")


# ============================================================
# CARICAMENTO MODELLO
# ============================================================

print("Caricamento modello...")
model = load_model(MODEL_ID)
print("Modello caricato.")


# ============================================================
# GENERAZIONE AUDIO
# ============================================================

all_audio_chunks = []
sample_rate = None

print(f"Blocchi testuali da generare: {len(text_chunks)}")

for idx, text_chunk in enumerate(text_chunks, start=1):
    print(f"\nGenero blocco {idx}/{len(text_chunks)}")
    print(f"Testo: {text_chunk}")

    # Generazione progressiva del blocco
    results = list(
        model.generate(
            text=text_chunk,
            ref_audio=REF_AUDIO_PATH,
            ref_text=ref_text,
        )
    )

    if not results:
        print(f"Attenzione: nessun risultato per il blocco {idx}")
        continue

    local_audio_chunks = []

    for result in results:
        chunk = np.asarray(result.audio)

        # Se il chunk ha dimensioni extra, le rimuove
        if chunk.ndim > 1:
            chunk = np.squeeze(chunk)

        # Tiene solo chunk validi
        if chunk.size > 0:
            local_audio_chunks.append(chunk.astype(np.float32))

        # Prende il sample rate dal primo risultato disponibile
        if sample_rate is None:
            sample_rate = getattr(result, "sample_rate", None)

    if not local_audio_chunks:
        print(f"Attenzione: nessun chunk audio valido per il blocco {idx}")
        continue

    # Unisce i chunk del singolo blocco
    block_audio = np.concatenate(local_audio_chunks).astype(np.float32)
    all_audio_chunks.append(block_audio)

    # Aggiunge una breve pausa, tranne dopo l'ultimo blocco
    if idx < len(text_chunks):
        if sample_rate is None:
            sample_rate = getattr(model, "sample_rate", 24000)

        silence = create_silence(PAUSE_MS_BETWEEN_CHUNKS, sample_rate)
        all_audio_chunks.append(silence)


# ============================================================
# CONTROLLO FINALE
# ============================================================

if not all_audio_chunks:
    raise RuntimeError("Il modello non ha prodotto audio valido.")

if sample_rate is None:
    sample_rate = getattr(model, "sample_rate", 24000)


# ============================================================
# UNIONE, NORMALIZZAZIONE E SALVATAGGIO
# ============================================================

final_audio = np.concatenate(all_audio_chunks)
final_audio = normalize_audio(final_audio, peak=0.95)

sf.write(OUTPUT_PATH, final_audio, sample_rate, subtype="PCM_16")


# ============================================================
# RIEPILOGO
# ============================================================

duration_sec = len(final_audio) / sample_rate

print("\nGenerazione completata.")
print(f"File generato: {OUTPUT_PATH}")
print(f"Blocchi testuali: {len(text_chunks)}")
print(f"Sample rate: {sample_rate}")
print(f"Durata approssimativa: {duration_sec:.2f} secondi")