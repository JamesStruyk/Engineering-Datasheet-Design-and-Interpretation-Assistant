
#beginning of RAG pipeline - extracting from PDF
#pip install pdfplumber sentence-transformers chromadb ollama --break-system-packages

import sys
import json
import os
import pdfplumber




#function for extracting page by page text and table data
def extract_datasheet(path):
    pages=[]

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):

            #get text
            text = page.extract_text() or ""

            #get tables
            tables = page.extract_tables()

            #add to pages dict
            pages.append({
                "pageNum": i+1,
                "text": text,
                "tables": tables,
            })
    return pages


def split_text(text, max_chars=800, overlap=120):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))

        # prefer to break on a newline in the back half of the window
        if end < len(text):
            nl = text.rfind("\n", start + max_chars // 2, end)
            if nl != -1:
                end = nl

        part = text[start:end].strip()
        if part:
            parts.append(part)

        if end >= len(text):
            break
        start = end - overlap   # step back so chunks overlap

    return parts

#function to turn pages to chunks
def chunk_datasheet(pages, datasheetName):
    chunks = []

    for p in pages:
        # text: split into smaller pieces
        for j, piece in enumerate(split_text(p["text"])):
            chunks.append({
                "id": f"{datasheetName}_p{p['pageNum']}_text{j}",
                "content": f"{datasheetName} | Page {p['pageNum']}\n{piece}",
                "page": p["pageNum"],
                "type": "text",
            })

        # tables
        for i, table in enumerate(p["tables"]):
            rowsToText = [" | ".join(str(cell) if cell is not None else "" for cell in row) for row in table]
            tableString = "\n".join(rowsToText)
            chunks.append({
                "id": f"{datasheetName}_p{p['pageNum']}_table{i}",
                "content": f"{datasheetName} | Page {p['pageNum']} | Table\n{tableString}",
                "page": p["pageNum"],
                "type": "table",
            })

    return chunks

def main():

    #make sure user gave path to datasheet pdf
    if len(sys.argv)<2:
        print("Usage: python extract.py <path_to_pdf>")
        sys.exit(1)

    pdfPath = sys.argv[1]

    datasheetName = os.path.splitext(os.path.basename(pdfPath))[0]
    pages = extract_datasheet(pdfPath)
    print(f"Extracted {len(pages)} from {pdfPath}")

    chunks = chunk_datasheet(pages, datasheetName)

    textChunkCount = sum(1 for chunk in chunks if chunk["type"]=="text")
    tableChunkCount = sum(1 for chunk in chunks if chunk["type"]=="table")

    print(f"  {len(chunks)} chunks created ({textChunkCount} text, {tableChunkCount} table)")

    #save chunk to JSON for next step in RAG
    outPath = f"chunks_{datasheetName}.json"
    with open(outPath, "w") as file:
        json.dump(chunks,file,indent=2)
    print(f"saved chunk data to {outPath}")


if __name__ == "__main__":
    main()