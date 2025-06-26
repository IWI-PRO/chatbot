import faiss
import numpy as np
import pickle
import os

class FAISSStore:
    def __init__(self, dim, index_file="faiss.index", metadata_file="metadata.pkl"):
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add_embeddings(self, embeddings, texts):
        self.index.add(np.array(embeddings).astype("float32"))
        self.metadata.extend(texts)
        self.save_index()

    def save_index(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, "rb") as f:
                self.metadata = pickle.load(f)

    def search(self, query_embedding, top_k=5):
        D, I = self.index.search(np.array([query_embedding]).astype("float32"), top_k)
        return [(self.metadata[i], float(D[0][j])) for j, i in enumerate(I[0])]
