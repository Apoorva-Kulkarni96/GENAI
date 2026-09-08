from retrieval.sparse.tokenizer import tokenize

class InvertedIndex:
    def __init__(self):
        self.index = {}
    def add(self,documents):
        for doc_id,document in enumerate(documents):
            terms = tokenize(document)
            for term in terms:
                if term in self.index:
                    self.index[term].append(doc_id)
                else:
                    self.index[term] = [doc_id]
        return self.index




