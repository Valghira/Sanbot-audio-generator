import csv
import json
import re
from pathlib import Path

from num2words import num2words


INPUT_JSON = Path("../testi/content.json")
OUTPUT_CSV = Path("../output/prepared_texts.csv")


def add_final_pause_marker(text):
    """Aggiunge un punto finale extra per migliorare la pausa finale."""
    return text.rstrip(".") + ".."


def normalize_spaces(text):
    """Rimuove spazi doppi e spazi inutili."""
    return re.sub(r"\s+", " ", text).strip()


def convert_numbers_to_words(text, lang="it"):
    """Converte i numeri interi presenti nel testo in lettere."""

    def replace_number(match):
        number = int(match.group())
        return num2words(number, lang=lang)

    return re.sub(r"\d+", replace_number, text)


def prepare_text(text, lang):
    """Applica le trasformazioni base al testo."""
    text = normalize_spaces(text)

    if lang == "it":
        text = convert_numbers_to_words(text, lang="it")

    text = add_final_pause_marker(text)
    return text


def extract_rows(data):
    """Estrae tutte le frasi dal file content.json."""
    rows = []
    counter = 1

    for lang, lang_data in data.items():
        for age_group, age_data in lang_data.items():
            for accessibility, sections in age_data.items():
                for section_index, section in enumerate(sections, start=1):
                    for title, sentences in section.items():
                        for sentence_index, original_text in enumerate(sentences, start=1):
                            prepared_text = prepare_text(original_text, lang)

                            rows.append({
                                "id": counter,
                                "lang": lang,
                                "age_group": age_group,
                                "accessibility": accessibility,
                                "section_index": section_index,
                                "title": title,
                                "sentence_index": sentence_index,
                                "original_text": original_text,
                                "prepared_text": prepared_text,
                                "status": "to_review",
                                "notes": ""
                            })

                            counter += 1

    return rows


def main():
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))

    rows = extract_rows(data)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"File generato: {OUTPUT_CSV}")
    print(f"Frasi estratte: {len(rows)}")


if __name__ == "__main__":
    main()