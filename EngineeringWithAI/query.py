
import sys
import argparse 
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

DB_PATH = "./datasheet_embeddings"
COLLECTION_NAME = "datasheet_chunks"

MODEL = "llama3.1"

embedder = None
collection = None

def get_embedder():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return embedder

def get_collection():
    global collection
    if collection is None:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_or_create_collection(COLLECTION_NAME)
    return collection

def retrieve(query,top_k=5):
    embedder = get_embedder()
    collection = get_collection()

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    hits = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        hits.append({
            "content": doc,
            "page": meta["page"],
            "type": meta["type"],
        })
    return hits

def answer_with_RAG(query, top_k=5):
    hits = retrieve(query, top_k=top_k)

    #print results to check what was retrieved
    for h in hits:
        preview = h["content"][:120].replace("\n", " ")
        print(f"[Page {h['page']}, {h['type']}] {preview}")

    context = "\n\n".join(f"[Page {h['page']}] {h['content']}" for h in hits)

    prompt = f"""
    You are an AI assistant helping to answer questions based on a datasheet. 
    Use the following context to answer the question below. 
    If the answer is not contained within the context, say you are unable to answer the question.
    Cite the page number(s) from the context where you found the answer.
    Context:
    {context}
    Question: 
    {query}
    """
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]

def answer_baseline(query, partName=""):

    partString = f" for part {partName}" if partName else ""
    prompt = f"""
    Answer this question {partString}: {query}
    """
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]

def main():
    parser = argparse.ArgumentParser(description="Answer questions based on a datasheet.")
    parser.add_argument("query", type=str, help="The question to ask.")
    parser.add_argument("--part", default="", type=str, help="The part name for baseline prompt.")
    parser.add_argument("--top_k", default=5, type=int, help="Number of top chunks to retrieve for RAG.")
    parser.add_argument("--baseline", action="store_true", help="Use baseline model instead of RAG." )
    args = parser.parse_args()

    if not args.baseline:
        print("Answering with RAG...")
        answer = answer_with_RAG(args.query, top_k=args.top_k)

    else:
        print("Answering with baseline model...")
        answer = answer_baseline(args.query, partName=args.part)

    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()