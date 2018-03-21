import numpy as np

PAD = "<PAD>"
UNK = "<UNK>"

class GloveEmbeddings:
    def __init__(self, file, additional_dim=0):
        self.embeddings = {}
        self.emb_size = None
        with open(file, encoding='utf8') as f:
            for i, line in enumerate(f):
                parts = line.split()
                cur_length = len(parts) - 1
                if self.emb_size is None:
                    self.emb_size = cur_length
                if cur_length != self.emb_size:
                    print(i + 1, line)
                    continue
                assert cur_length == self.emb_size

                word = parts[0]
                emb_arr = [float(e) for e in parts[1:]]
                if additional_dim > 0:
                    for i in range(additional_dim):
                        emb_arr.append(0.0)
                emb = np.array(emb_arr)

                self.embeddings[word] = emb
        self.emb_size += additional_dim
        self.unk_embedding = np.zeros(self.emb_size)
        self.pad_embedding = np.zeros(self.emb_size)

    # this function must be implemented for embeddings
    def get_embedding(self, word):
        if word in self.embeddings:
            return self.embeddings[word]
        elif word == PAD:
            return self.pad_embedding
        else:
            word = UNK
            return self.unk_embedding