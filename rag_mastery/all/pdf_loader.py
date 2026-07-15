from pathlib import Path
import pymupdf # type: ignore
import sys

#pdf_path = Path(r"C:\Users\apoor\OneDrive\Desktop\git\GENAI\rag_mastery\datasets\raw\papers\RAG.pdf")

class PDFLoader():
    def __init__(self):
        pass

    def load(self, pdf_path):
        pages = []
        with pymupdf.open(pdf_path) as doc:
            for page in doc:
                pages.append(page.get_text())
        return "".join(pages)









