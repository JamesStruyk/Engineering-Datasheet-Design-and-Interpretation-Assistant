

#Store pdf chunks as embeddings to be searched later
import sys
import json
from sentence_transformers import SentenceTransformer
import chromadb

DB_PATH = "./datasheet_embeddings" 
COLLECTION_NAME = "datasheet_chunks"

def load_chunks(path):
    with open(path, "r") as f:
        chunks = json.load(f)
    return chunks

def main():
    if len(sys.argv)<2:
        print("Usage: python index.py <chunks.json>")
        sys.exit(1)

    chunksPath = sys.argv[1]
    chunks = load_chunks(chunksPath)
    print(f"Loaded {len(chunks)} chunks from {chunksPath}")

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path=DB_PATH)
    try: 
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(COLLECTION_NAME)
    chunks = [c for c in chunks if c["content"].strip()]
    contents = [c["content"] for c in chunks]
    embeddings = embedder.encode(contents, show_progress_bar=True).tolist()    

    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=contents,
        metadatas=[{"page": c["page"], "type": c["type"]} for c in chunks],
    )

    print(f"Indexed {len(chunks)} chunks into collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()