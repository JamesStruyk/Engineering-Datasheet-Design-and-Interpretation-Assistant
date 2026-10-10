
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

#function to turn pages to chunks
def chunk_datasheet(pages,datasheetName):
    chunks = []

    for p in pages:
        if p["text"].strip():
            chunks.append({
                "id": f"{datasheetName}_p{p['pageNum']}_text",
                "content": p["text"],
                "page": p["pageNum"],
                "type": "text",
            })

        for i, table in enumerate(p["tables"]):

            rowsToText = [ " | ".join(str(cell) if cell is not None else "" for cell in row) for row in table]
            tableString = "\n".join(rowsToText)
            chunks.append({
                "id": f"{datasheetName}_p{p['pageNum']}_table{i}",
                "content": tableString,
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