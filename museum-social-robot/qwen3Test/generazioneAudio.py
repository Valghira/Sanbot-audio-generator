from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

# Path serve per leggere il file di testo in modo semplice e robusto.
# numpy viene usato per manipolare i campioni audio come array numerici.
# soundfile salva l'audio finale su disco in formato WAV.
# load_model carica il modello TTS MLX da usare per la generazione.


# MODEL_ID è lasciato come promemoria, ma sotto il modello viene caricato direttamente tramite stringa.
# Audio di riferimento: contiene la voce da imitare.
REF_AUDIO_PATH = "cutdonna.wav"
# Trascrizione dell'audio di riferimento: serve al modello per allineare voce e contenuto.
REF_TEXT_PATH = "testodonna.txt"
# Nome del file audio finale che verrà generato.
OUTPUT_PATH = "outfile.wav"
# Testo che il modello dovrà leggere con la voce clonata/adattata dal riferimento.
TEXT_TO_READ = ("Benvenuti al museo della sindone di Torino! Oggi scopriremo insieme la storia della sindone e del museo. Io sarò la vostra guida. Siete pronti?")


# Legge il testo associato all'audio di riferimento.
# .strip() rimuove spazi e righe vuote iniziali/finali.
ref_text = Path(REF_TEXT_PATH).read_text(encoding="utf-8").strip()

# Se il file è vuoto, il modello non ha un riferimento testuale affidabile.
if not ref_text:
    raise RuntimeError(f"Il file di testo {REF_TEXT_PATH} è vuoto.")

# Carica il modello text-to-speech.
model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")

# Avvia la generazione audio.
# Il modello riceve:
# - text: il testo da pronunciare
# - ref_audio: il file audio con la voce di riferimento
# - ref_text: la trascrizione del file audio di riferimento
# model.generate restituisce risultati progressivi; qui li trasformiamo subito in lista.
results = list(model.generate(
    text=TEXT_TO_READ,
    ref_audio=REF_AUDIO_PATH,
    ref_text=ref_text,
))

# Controllo di sicurezza: se non arriva alcun risultato, interrompiamo lo script.
if not results:
    raise RuntimeError("Nessun risultato generato dal modello.")

# Qui raccogliamo i pezzi audio generati dal modello.
# "chunks" = piccoli segmenti di audio che poi verranno uniti.
chunks = []
sample_rate = None

for result in results:
    # Converte l'audio del risultato in array numpy.
    chunk = np.asarray(result.audio)

    # Se l'array ha dimensioni extra (es. 1 x N), le rimuove.
    if chunk.ndim > 1:
        chunk = np.squeeze(chunk)

    # Salva solo i chunk effettivamente non vuoti.
    if chunk.size > 0:
        chunks.append(chunk.astype(np.float32))

    # Recupera il sample rate dal primo risultato disponibile.
    if sample_rate is None:
        sample_rate = getattr(result, "sample_rate", None)

# Se non abbiamo nessun chunk valido, non possiamo costruire l'audio finale.
if not chunks:
    raise RuntimeError("Il modello non ha prodotto chunk audio validi.")

# Se il sample rate non è stato trovato nei risultati, prova a prenderlo dal modello.
# Se manca anche lì, usa 24000 Hz come valore di fallback.
if sample_rate is None:
    sample_rate = getattr(model, "sample_rate", 24000)

# Unisce tutti i chunk in un unico array audio continuo.
audio = np.concatenate(chunks)

# Calcola il valore assoluto massimo del segnale per normalizzarlo.
max_abs = float(np.max(np.abs(audio)))
if max_abs > 0:
    # Normalizzazione: evita clipping e porta il volume vicino al massimo senza saturare.
    audio = audio / max_abs * 0.95

# Converte definitivamente l'audio in float32.
audio = audio.astype(np.float32)

# Salva il file WAV finale.
# PCM_16 indica un formato WAV standard a 16 bit, molto compatibile.
sf.write(OUTPUT_PATH, audio, sample_rate, subtype="PCM_16")

# Messaggi finali di riepilogo utili per controllare il risultato.
print(f"File generato: {OUTPUT_PATH}")
print(f"Chunk audio generati: {len(chunks)}")
print(f"Sample rate: {sample_rate}")
print(f"Durata approssimativa: {len(audio) / sample_rate:.2f} secondi")