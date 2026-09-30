import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model


# ============================================================
# CONFIGURAZIONE
# ============================================================

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

INPUT_CSV = Path("../output/prepared_texts.csv")
OUTPUT_DIR = Path("../output/audio")

REF_AUDIO_PATH = "../../cutdonna.wav"
REF_TEXT_PATH = "../../testodonna.txt"

# Scegli da dove iniziare e dove fermarti
START_ID = 1
END_ID = 2
# Colonna del CSV da usare per generare l'audio
TEXT_COLUMN = "prepared_text"


def build_title_indexes(rows):
    """
    Ricostruisce l'indice della chiave/titolo dentro ogni blocco del JSON.

    Naming desiderato:
    audio_<indice_blocco>_<indice_titolo>_<indice_frase>.wav

    Gli indici del JSON partono da 0, quindi:
    - section_index del CSV viene convertito da 1-based a 0-based
    - sentence_index del CSV viene convertito da 1-based a 0-based
    - title_index viene calcolato in base all'ordine di apparizione dei titoli nel CSV
    """
    title_indexes = {}
    next_title_index = {}

    for row in rows:
        group_key = (
            row["lang"],
            row["age_group"],
            row["accessibility"],
            row["section_index"],
        )
        title_key = (*group_key, row["title"])

        if group_key not in next_title_index:
            next_title_index[group_key] = 0

        if title_key not in title_indexes:
            title_indexes[title_key] = next_title_index[group_key]
            next_title_index[group_key] += 1

    return title_indexes


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ref_text = Path(REF_TEXT_PATH).read_text(encoding="utf-8").strip()

    if not ref_text:
        raise RuntimeError(f"Il file {REF_TEXT_PATH} è vuoto.")

    print("Caricamento modello...")
    model = load_model(MODEL_ID)

    with INPUT_CSV.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    # Calcola gli indici dei titoli usando tutte le righe del CSV,
    # non solo quelle selezionate, così il nome resta coerente con la posizione nel JSON.
    title_indexes = build_title_indexes(rows)

    selected_rows = [
        row for row in rows
        if START_ID <= int(row["id"]) <= END_ID
    ]

    print(f"Frasi selezionate: {len(selected_rows)}")
    print(f"Intervallo ID: {START_ID} - {END_ID}")

    for row in selected_rows:
        row_id = int(row["id"])
        lang = row["lang"]
        age_group = row["age_group"]
        accessibility = row["accessibility"]
        section_index = int(row["section_index"]) - 1
        sentence_index = int(row["sentence_index"]) - 1
        title = row["title"]
        text = row[TEXT_COLUMN].strip()

        title_key = (
            lang,
            age_group,
            accessibility,
            row["section_index"],
            title,
        )
        title_index = title_indexes[title_key]

        if not text:
            print(f"[SKIP] ID {row_id}: testo vuoto")
            continue

        # Nome compatibile con la struttura annidata del JSON:
        # audio_<indice_blocco>_<indice_titolo>_<indice_frase>.wav
        output_filename = f"audio_{section_index}_{title_index}_{sentence_index}.wav"

        # Le sottocartelle evitano sovrascritture tra italiano/inglese,
        # adult/child e typical/blind/deaf, dato che i nomi audio si ripetono.
        output_path = OUTPUT_DIR / lang / age_group / accessibility / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("\n====================================")
        print(f"ID: {row_id}")
        print(f"Titolo: {title}")
        print(f"Nome file: {output_filename}")
        print(f"Output: {output_path}")
        print("Testo:")
        print(text)

        audio, sample_rate = generate_audio(
            model=model,
            text=text,
            ref_audio_path=REF_AUDIO_PATH,
            ref_text=ref_text,
        )

        sf.write(output_path, audio, sample_rate, subtype="PCM_16")

        print(f"Generato: {output_path}")
        print(f"Durata: {len(audio) / sample_rate:.2f} secondi")

    print("\nGenerazione completata.")


if __name__ == "__main__":
    main()