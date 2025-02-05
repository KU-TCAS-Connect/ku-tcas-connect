from FlagEmbedding import BGEM3FlagModel

class FlagModel:
    def __init__(self):
        self.bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)