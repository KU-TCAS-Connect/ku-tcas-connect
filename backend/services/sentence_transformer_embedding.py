from sentence_transformers import SentenceTransformer


class SentenceTransformerModel:
    def __init__(self):
        self.st_paraphrase = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")