import json
from pathlib import Path
from collections import defaultdict

# load dataset
def load_hotpot(path):
    with open(path, "r") as f:
        return json.load(f)
    

# Extract and deduplicate documents

def extract_documents(hotpot_data):
    """
    Extract unique documents from HotpotQA
    Key = title
    Value = concatenated text
    """

    documents = {}

    for example in hotpot_data:
        for title, sentences in example['context']:
            if title not in documents:
                documents[title] = " ".join(sentences)

    return documents

# save documents as json

def save_documents(documents, output_path):

    output_path.parent.mkdir(parents = True, exist_ok = True)

    with open(output_path, "w") as f:
        for title, text in documents.items():
            f.write(json.dumps(
                {
                    "title" : title,
                    "text" : text
                }
            ) + "\n")


# document chunking

def chunk_text(text, max_tokens=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)

    return chunks


# create chuncking with metadata
def create_chunks(documents):
    chunks = []

    for title, text in documents.items():
        for idx, chunk in enumerate(chunk_text(text)):
            chunks.append({
                "doc_title": title,
                "chunk_id": idx,
                "text": chunk
            })

    return chunks

# save chunking

def save_chunks(chunks, output_path):
    with open(output_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

def print_stats(documents, chunks):
    print("Number of unique documents:", len(documents))
    print("Number of chunks:", len(chunks))

    lengths = [len(d.split()) for d in documents.values()]
    print("Avg document length:", sum(lengths) / len(lengths))


def main():
    data_path = "./data/hotpot_dev_distractor_v1.json"
    out_docs = Path("processed/documents.jsonl")
    out_chunks = Path("processed/chunks.jsonl")

    hotpot = load_hotpot(data_path)
    documents = extract_documents(hotpot)
    save_documents(documents, out_docs)

    chunks = create_chunks(documents)
    save_chunks(chunks, out_chunks)

    print_stats(documents, chunks)

if __name__ == "__main__":
    main()
