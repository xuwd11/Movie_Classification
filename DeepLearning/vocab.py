import numpy as np
from string import punctuation
import pickle

class Vocab:
    def __init__(self, lower=True, strip=True, remove_punctuations=True):
        self.embs = None
        self.emb_size = None
        self.word_to_ind = {None:0}
        self.ind_to_word = [None]
        self.lower = lower
        self.strip = strip
        self.remove_punctuations = remove_punctuations
        self._initialized = False
    
    def process_word(self, word):
        if self.lower:
            word = word.lower()
        if self.strip:
            word = word.strip()
        if self.remove_punctuations:
            word = word.strip(punctuation)
        return word
    
    def initialize_glove(self, glove):
        assert(not self._initialized)
        if isinstance(glove, str):
            return self.initialize_glove(open(glove, 'r').readlines())
        emb_size = len(glove[0].split())-1
        self.emb_size = emb_size
        self.embs = [np.zeros(emb_size)]
        for emb in glove:
            parts = emb.split()
            assert(len(parts)==emb_size+1)
            w = parts[0].strip()
            if w in self.word_to_ind:
                print(parts[0])
            assert(w not in self.word_to_ind)
            self.word_to_ind[w] = len(self.ind_to_word)
            self.ind_to_word.append(w)
            self.embs.append(np.array(parts[1:]))
        self._initialized = True
        
    def add_sentences(self, sentences, new_word='zero'):
        new_words_arr = []
        for s in sentences:
            if isinstance(s, np.ndarray) or isinstance(s, list):
                for ss in s:
                    new_words_arr.append(self.add_sentence(ss, new_word))
            else:
                new_words_arr.append(self.add_sentence(s, new_word))
        return np.concatenate(new_words_arr)
    
    def add_sentence(self, sentence, new_word='zero'):
        assert isinstance(sentence, str)
        assert(new_word in {'zero', 'random'})
        new_words = []
        for word in sentence.split():
            if word in self.word_to_ind:
                continue
            w = self.process_word(word)
            if w in self.word_to_ind or len(w)==0:
                continue
            else:
                if new_word == "zero":
                    emb = np.zeros(self.emb_size)
                else:
                    emb = np.random.rand(self.emb_size)
                self.word_to_ind[w] = len(self.ind_to_word)
                self.ind_to_word.append(w)
                self.embs.append(emb)
                new_words.append(w)
        return np.array(new_words)
    
    def __len__(self):
        return len(self.ind_to_word)
    
    def save(self, file):
        pickle.dump(self, open(file, "wb"))
    
    def get_embedding_matrix(self):
        return np.array(self.embs, dtype="double")
    
    def encode_sentences(self, sentences):
        if isinstance(sentences[0], np.ndarray) or isinstance(sentences[0], list):
            result = np.empty(len(sentences), dtype=object)
            for d,s in enumerate(sentences):
                dr = np.empty(len(s), dtype=object)
                for i,ss in enumerate(s):
                    dr[i] = self.encode_sentence(ss)
                result[d] = dr
        else:
            result = np.empty(len(sentences), dtype=object)
            for i,s in enumerate(sentences):
                result[i] = self.encode_sentence(s)
        return result
    
    def encode_sentence(self, sentence):
        assert isinstance(sentence, str)
        result = []
        for word in sentence.split():
            if word in self.word_to_ind:
                result.append(self.word_to_ind[word])
            else:
                w = self.process_word(word)
                if w in self.word_to_ind:
                    result.append(self.word_to_ind[w])
                elif len(w):
                    result.append(0)
        return np.array(result)
    
    def decode_sentence(self, sentence_encoded):
        result = []
        for e in sentence_encoded:
            if e == 0:
                result.append("UNK")
            else:
                result.append(self.ind_to_word[e])
        return " ".join(result)
    
    def create_pytorch_embeddings(self, freeze=True):
        import torch
        embeddings = self.get_embedding_matrix()
        rows, cols = embeddings.shape
        embedding = torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols, padding_idx=0)
        embedding.weight.data.copy_(torch.from_numpy(embeddings))
        embedding.weight.requires_grad = not freeze
        return embedding


def load_vocab(file):
    return pickle.load(open(file, 'rb'))