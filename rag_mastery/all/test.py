from pdf_loader import PDFLoader

loader = PDFLoader()
text = loader.load(r"C:\Users\apoor\OneDrive\Desktop\git\GENAI\rag_mastery\datasets\raw\papers\RAG.pdf")
print(text)


print(type(text))
print(len(text))

print(text[0:1])