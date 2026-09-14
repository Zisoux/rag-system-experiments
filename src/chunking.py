from pathlib import Path


def load_document(file_path):
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        text = file.read()

    return text


def split_into_chunks(text):
    chunks = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    return chunks


if __name__ == "__main__":
    document = load_document("data/sample.txt")
    chunks = split_into_chunks(document)

    for index, chunk in enumerate(chunks, start=1):
        print(f"[Chunk {index}]")
        print(chunk)
        print()

