from chunkers.text_chunker import TextChunker
from  llms.ollama_llm import OllamaLLM
from loaders.pdf_loader import PDFLoader
from retrievers.retriever import Retriever
from pipelines.rag_pipeline import RAGPipeline
from prompt.prompt_builder import PromptBuilder
from sentence_transformers import SentenceTransformer
from pathlib import Path

pdf_path = Path(r"C:\Users\apoor\OneDrive\Desktop\git\GENAI\rag_mastery\datasets\raw\papers\RAG.pdf")
model = SentenceTransformer('all-MiniLM-L6-v2')
llm_model = "llama3.1:latest"

query = "What is RAG?"
k=3


loader = PDFLoader()
text = loader.load(pdf_path)


chunker = TextChunker()
chunks=chunker.split(text)

retriever = Retriever(model)
retriever.fit(model)

prompt_builder = PromptBuilder()
llm = OllamaLLM()

pipeline = RAGPipeline(
    retriever,
    prompt_builder,
    llm, 
)

response=pipeline.ask(query,k,llm_model)
print(response)