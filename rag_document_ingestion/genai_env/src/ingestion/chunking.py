from sentence_transformers import SentenceTransformer
import numpy as np
import re

class TextChunker:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    

    def extract_sections(self, text):
        sections = []
        current_section = None

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            if re.match(r"^\d+\.\s+.+", line):
                current_section = line
                continue

            sections.append({
                "text": line,
                "section": current_section
            })

        return sections

    # -------------------------
    # 1. Fixed-size chunking
    # -------------------------
    def fixed_chunk(self, text, chunk_size=500, overlap=50):
        chunks = []

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])

            start += chunk_size - overlap

        return chunks

    # -------------------------
    # 2. Recursive chunking
    # -------------------------
    def recursive_chunk(
        self,
        text,
        chunk_size=500,
        overlap=50,
        separators=None
    ):
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]

        # For learning purposes, use LangChain's implementation
        # rather than rebuilding the recursive algorithm.
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=separators
        )

        return splitter.split_text(text)

    def recursive_chunk_with_sections(self, text, chunk_size=500, overlap=50):
        sections = self.extract_sections(text)

        chunks = []
        current_text = ""
        current_section = None

        for item in sections:
            line = item["text"]
            section = item["section"]

            if current_section is None:
                current_section = section

            # Section changed
            if section != current_section:
                section_chunks = self.recursive_chunk(
                    current_text,
                    chunk_size,
                    overlap
                )

                for chunk in section_chunks:
                    chunks.append({
                        "text": chunk,
                        "section": current_section
                    })

                current_text = ""
                current_section = section

            current_text += line + "\n"

        # Process last section
        if current_text.strip():
            section_chunks = self.recursive_chunk(
                current_text,
                chunk_size,
                overlap
            )

            for chunk in section_chunks:
                chunks.append({
                    "text": chunk,
                    "section": current_section
                })

        return chunks

    # -------------------------
    # 3. Semantic chunking
    # -------------------------
    def semantic_chunk(self, text, threshold=0.5):
        sentences = text.split(". ")

        if not sentences:
            return []


        embeddings = self.model.encode(sentences)

        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):

            previous = embeddings[i - 1]
            current = embeddings[i]

            similarity = np.dot(previous, current) / (
                np.linalg.norm(previous) *
                np.linalg.norm(current)
            )

            if similarity < threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        chunks.append(" ".join(current_chunk))

        return chunks
        # -------------------------
    # 4. Parent-Document / Small-to-Big
    # -------------------------
    def parent_child_chunk(
        self,
        text,
        parent_size=1000,
        child_size=300,
        overlap=50
    ):
        parents = self.recursive_chunk(
            text,
            chunk_size=parent_size,
            overlap=0
        )

        children = []

        for parent_id, parent in enumerate(parents):

            child_chunks = self.recursive_chunk(
                parent,
                chunk_size=child_size,
                overlap=overlap
            )

            for child in child_chunks:
                children.append({
                    "text": child,
                    "parent_id": parent_id,
                    "parent_text": parent
                })

        return parents, children



