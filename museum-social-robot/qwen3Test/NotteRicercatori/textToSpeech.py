import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model


# ============================================================
# CONFIGURAZIONE
# ============================================================

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

BASE_DIR = Path(__file__).parent

INPUT_CSV = BASE_DIR / "frasi.csv"
OUTPUT_DIR = BASE_DIR / "output"

REF_AUDIO_PATH = BASE_DIR.parent / "cutdonna.wav"
REF_TEXT_PATH = BASE_DIR.parent / "testodonna.txt"

# Scegli da dove iniziare e dove fermarti
START_ID = 4
END_ID = 4

# Colonna del CSV da usare per generare l'audio
TEXT_COLUMN = "modified_text"


# ============================================================
# FUNZIONI AUDIO
# ============================================================

def generate_audio(model, text, ref_audio_path, ref_text):
    results = list(model.generate(
        text=text,
        ref_audio=ref_audio_path,
        ref_text=ref_text,
    ))

    if not results:
        raise RuntimeError("Nessun risultato generato dal modello.")

    chunks = []
    sample_rate = None

    for result in results:
        chunk = np.asarray(result.audio)

        if chunk.ndim > 1:
            chunk = np.squeeze(chunk)

        if chunk.size > 0:
            chunks.append(chunk.astype(np.float32))

        if sample_rate is None:
            sample_rate = getattr(result, "sample_rate", None)

    if not chunks:
        raise RuntimeError("Il modello non ha prodotto audio valido.")

    if sample_rate is None:
        sample_rate = getattr(model, "sample_rate", 24000)

    audio = np.concatenate(chunks)

    max_abs = float(np.max(np.abs(audio)))
    if max_abs > 0:
        audio = audio / max_abs * 0.95

    return audio.astype(np.float32), sample_rate


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ref_text = REF_TEXT_PATH.read_text(encoding="utf-8").strip()

    if not ref_text:
        raise RuntimeError(f"Il file {REF_TEXT_PATH} è vuoto.")

    print("Caricamento modello...")
    model = load_model(MODEL_ID)

    with INPUT_CSV.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    selected_rows = [
        row for row in rows
        if START_ID <= int(row["id"]) <= END_ID
    ]

    print(f"Frasi selezionate: {len(selected_rows)}")
    print(f"Intervallo ID: {START_ID} - {END_ID}")

    for row in selected_rows:
        row_id = int(row["id"])
        text = row[TEXT_COLUMN].strip()

        if not text:
            print(f"[SKIP] ID {row_id}: testo vuoto")
            continue

        # ID 1 -> 001.wav, ID 2 -> 002.wav, ..., ID 14 -> 014.wav
        output_filename = f"{row_id:03d}.wav"
        output_path = OUTPUT_DIR / output_filename

        print("\n====================================")
        print(f"ID: {row_id}")
        print(f"Nome file: {output_filename}")
        print("Testo:")
        print(text)

        audio, sample_rate = generate_audio(
            model=model,
            text=text,
            ref_audio_path=str(REF_AUDIO_PATH),
            ref_text=ref_text,
        )

        sf.write(
            output_path,
            audio,
            sample_rate,
            subtype="PCM_16"
        )

        print(f"Generato: {output_path}")
        print(f"Durata: {len(audio) / sample_rate:.2f} secondi")

    print("\nGenerazione completata.")


if __name__ == "__main__":
    main()