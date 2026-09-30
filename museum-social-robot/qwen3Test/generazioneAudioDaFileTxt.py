from pathlib import Path
import logging
import re

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging


# ============================================================
# CONFIGURAZIONE
# ============================================================

# File audio di riferimento: contiene la voce da imitare/adattare
REF_AUDIO_PATH = "cutdonna.wav"

# Trascrizione esatta dell'audio di riferimento
REF_TEXT_PATH = "testodonna.txt"

# File di testo con il contenuto da leggere (input principale)
INPUT_TEXT_PATH = "testo_da_leggere.txt"

# File WAV finale generato
OUTPUT_PATH = "migliorato_da_file2.wav"

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
# LETTURA TESTI DI INPUT
# ============================================================

# Testo di riferimento usato dal modello per imitare lo stile/voce.
ref_text = Path(REF_TEXT_PATH).read_text(encoding="utf-8").strip()
if not ref_text:
    raise RuntimeError(f"Il file di testo {REF_TEXT_PATH} è vuoto.")
ref_text = normalize_text(ref_text)

# Testo principale da leggere: viene caricato da file .txt.
text_to_read = Path(INPUT_TEXT_PATH).read_text(encoding="utf-8").strip()
if not text_to_read:
    raise RuntimeError(f"Il file di input {INPUT_TEXT_PATH} è vuoto.")


# ============================================================
# PREPARAZIONE TESTO DA LEGGERE
# ============================================================

clean_text = normalize_text(text_to_read)

if not clean_text:
    raise RuntimeError("Il testo da leggere è vuoto dopo la normalizzazione.")

text_chunks = split_text_into_chunks(clean_text, max_chars=MAX_CHARS_PER_CHUNK)

if not text_chunks:
    raise RuntimeError("Non sono stati creati blocchi testuali validi.")


# ============================================================
# CARICAMENTO MODELLO
# ============================================================

print("Caricamento modello...")
# Riduce warning verbosi di Transformers durante il bootstrap del modello.
transformers_logging.set_verbosity_error()
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
model = load_model(MODEL_ID)

# Ricarica il tokenizer con fix_mistral_regex=True per evitare tokenizzazione
# non corretta con alcuni tokenizer Mistral/Qwen segnalati da Transformers.
try:
    tokenizer_source = getattr(getattr(model, "config", None), "model_path", MODEL_ID)
    model.tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        fix_mistral_regex=True,
        trust_remote_code=True,
    )
    print("Tokenizer ricaricato con fix_mistral_regex=True.")
except Exception as exc:
    print(f"Attenzione: impossibile ricaricare il tokenizer con fix regex: {exc}")

# Ripristina la verbosità standard dopo il caricamento completo.
transformers_logging.set_verbosity_warning()

print("Modello caricato.")


# ============================================================
# GENERAZIONE AUDIO
# ============================================================

# Nota: in questo script NON viene usato GPT.
# Il testo normalizzato del file .txt viene inviato direttamente al modello Qwen TTS.
all_audio_chunks = []
sample_rate = None

print(f"Blocchi testuali da generare: {len(text_chunks)}")

for idx, text_chunk in enumerate(text_chunks, start=1):
    print(f"\nGenero blocco {idx}/{len(text_chunks)}")
    print(f"Testo: {text_chunk}")

    # Generazione progressiva del blocco corrente:
    # model.generate restituisce più risultati/chunk che poi uniamo.
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

        # Se il chunk ha dimensioni extra, le rimuove.
        if chunk.ndim > 1:
            chunk = np.squeeze(chunk)

        # Tiene solo chunk validi.
        if chunk.size > 0:
            local_audio_chunks.append(chunk.astype(np.float32))

        # Prende il sample rate dal primo risultato disponibile.
        if sample_rate is None:
            sample_rate = getattr(result, "sample_rate", None)

    if not local_audio_chunks:
        print(f"Attenzione: nessun chunk audio valido per il blocco {idx}")
        continue

    # Unisce i chunk del singolo blocco in una sequenza continua.
    block_audio = np.concatenate(local_audio_chunks).astype(np.float32)
    all_audio_chunks.append(block_audio)

    # Inserisce una pausa breve tra blocchi per una resa più naturale
    # e per evitare che le frasi risultino "attaccate".
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

# Unisce tutti i blocchi (frasi + pause), normalizza il volume
# e salva il risultato finale in WAV 16-bit.
final_audio = np.concatenate(all_audio_chunks)
final_audio = normalize_audio(final_audio, peak=0.95)

sf.write(OUTPUT_PATH, final_audio, sample_rate, subtype="PCM_16")


# ============================================================
# RIEPILOGO
# ============================================================

duration_sec = len(final_audio) / sample_rate

print("\nGenerazione completata.")
print(f"File di testo letto: {INPUT_TEXT_PATH}")
print(f"File generato: {OUTPUT_PATH}")
print(f"Blocchi testuali: {len(text_chunks)}")
print(f"Sample rate: {sample_rate}")
print(f"Durata approssimativa: {duration_sec:.2f} secondi")
