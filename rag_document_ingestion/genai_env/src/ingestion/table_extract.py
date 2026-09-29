import pdfplumber


class TableExtractor:

    def clean_row(self, row):
        return [cell if cell is not None else "" for cell in row]

    def extract_tables(self, pdf_path):
        all_tables = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()

                for table in tables:
                    all_tables.append(table)

        return all_tables

    def table_to_markdown(self, table):
        if not table:
            return ""

        header = self.clean_row(table[0])

        markdown = "| " + " | ".join(header) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"

        for row in table[1:]:
            row = self.clean_row(row)
            markdown += "| " + " | ".join(row) + " |\n"

        return markdown


if __name__ == "__main__":

    extractor = TableExtractor()

    tables = extractor.extract_tables(r"C:\Users\apoor\OneDrive\Desktop\git\GENAI\rag_document_ingestion\genai_env\data\pdfs\BERT.pdf")
    for table in tables:
        print(extractor.table_to_markdown(table))
        print("=" * 70)