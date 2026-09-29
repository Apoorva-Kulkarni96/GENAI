import fitz


class PDFLoader():
    def load_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        content = []
        x_positions = []
        for page in doc:
            left_column = []
            right_column = []
            
            blocks = page.get_text("blocks")
            boundary = self.find_column_boundary(blocks)

            for block in blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                if boundary is not None and x0 < boundary:
                    left_column.append((y0, text))
                else:
                    right_column.append((y0, text))

            left_column.sort(key=lambda x: x[0])
            right_column.sort(key= lambda x :x[0])
            for y, text in left_column:
                content.append(text)
            for y, text in right_column:
                content.append(text)
        doc.close()
        return content

    def find_column_boundary(self, blocks):
        x_positions = []

        for block in blocks:
            x_positions.append(round(block[0], 1))

        x_positions = sorted(set(x_positions))

        for i in range(len(x_positions)-1):
            gap = x_positions[i+1] - x_positions[i]

            if gap > 20:
                boundary = (x_positions[i+1] + x_positions[i])/2
                return boundary
        return None

                

if __name__ == "__main__":
    pdf = PDFLoader()
    text = pdf.load_pdf(r"C:\Users\apoor\OneDrive\Desktop\git\GENAI\rag_document_ingestion\genai_env\data\pdfs\BERT.pdf")
    print(text)
