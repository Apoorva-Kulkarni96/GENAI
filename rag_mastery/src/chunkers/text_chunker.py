
class TextChunker:
    def __init__(self, chunk_size : int = 5000, overlap : int = 2000, separator : list[str] | None = None):
        

        if separator is None:
            separator = ["\n\n", "\n", ". ", " ",""]

        self.chunk_size = chunk_size
        self.separator = separator
        self.overlap = overlap


    def split(self, text):
        chunk = []
        step = self.chunk_size - self.overlap
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap should be smaller than chunk_size")
        for i in range(0, len(text),step):
            chunk.append(text[i:i+self.chunk_size])
        return chunk 
                

