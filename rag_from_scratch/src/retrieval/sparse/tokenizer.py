
def tokenize(text):
    return text.lower().split()


if __name__ == "__main__":
    text = "I love FAISS Search via Indexing"
    result = tokenize(text)
    print(result)