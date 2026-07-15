from chunker import TextChunker

chunk = TextChunker(chunk_size=5, overlap=2, separator=None)

chunks = chunk.split("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print(chunks)