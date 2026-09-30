import csv
import re
from pathlib import Path

from pypdf import PdfReader


PDF_PATH = Path(__file__).parent / "Frasi.pdf"
CSV_PATH = Path(__file__).parent / "frasi.csv"

START_MARKER = "Perfetto, hai scelto"


def extract_text_from_pdf():
    reader = PdfReader(PDF_PATH)

    pages = []

    for page in reader.pages:
        text = page.extract_text()
        pages.append(text)

    return "\n".join(pages)


def clean_extracted_text(text):
    # Sostituisce sequenze di spazi, newline, ecc. con un singolo spazio
    return re.sub(r"\s+", " ", text).strip()


def extract_sentences(text):
    # Elimina tutta la consegna precedente alla prima frase
    first_sentence = text.find(START_MARKER)

    if first_sentence == -1:
        raise ValueError(
            f'Non è stata trovata nessuna frase che inizi con "{START_MARKER}".'
        )

    text = text[first_sentence:]

    # Divide il testo ogni volta che incontra l'inizio di una nuova frase
    parts = text.split(START_MARKER)

    sentences = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        sentence = f"{START_MARKER} {part}".strip()
        sentences.append(sentence)

    return sentences


def prepare_modified_text(original_text):
    # Aggiunge un secondo punto finale per migliorare la pausa del TTS
    if original_text.endswith("."):
        return original_text + "."

    return original_text + ".."


def save_csv(sentences):
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["id", "original_text", "modified_text", "status"]
        )

        writer.writeheader()

        for sentence_id, original_text in enumerate(sentences, start=1):
            writer.writerow({
                "id": sentence_id,
                "original_text": original_text,
                "modified_text": prepare_modified_text(original_text),
                "status": "TO_REVIEW"
            })


def main():
    text = extract_text_from_pdf()
    text = clean_extracted_text(text)

    sentences = extract_sentences(text)

    print(f"Frasi trovate: {len(sentences)}")

    if len(sentences) != 14:
        raise ValueError(
            f"Attenzione: erano attese 14 frasi, ma ne sono state trovate {len(sentences)}."
        )

    save_csv(sentences)

    print(f"CSV creato: {CSV_PATH}")


if __name__ == "__main__":
    main()