import numpy as np
import math
import pickle
import torch
from string import punctuation
import pickle

class Data:
    def __init__(self, data:np.ndarray):
        self.data = data
    
    def create_splits(self, train_p, val_p=0, test_p=None, shuffle=True, random_seed=None):
        if isinstance(train_p, float):
            assert(isinstance(val_p, float) or val_p==0)
            
            train_n = int(len(self.data)*train_p)
            val_n = int(len(self.data)*val_p)
            
            remain_n = len(self.data)-train_n-val_n
            remain_p = 1-train_p-val_p
            assert(remain_p>=0)
            
            if test_p is not None:
                assert(isinstance(test_p, float))
                assert(test_p <= remain_p)
                test_n = min(remain_n, int(len(self.data)*test_p))
            else:
                test_n = remain_n
        elif isinstance(train_p, int):
            assert(isinstance(val_p, int))
            train_n = train_p
            val_n = val_p
            remain_n = len(self.data)-train_n-val_n
            assert(remain_n>=0)
            
            if test_p is not None:
                assert(isinstance(test_p, int))
                assert(test_p <= remain_n)
                test_n = test_p
            else:
                test_n = remain_n
        else:
            raise ValueError
        inds = np.array(list(range(len(self.data))))
        if shuffle:
            if random_seed is not None:
                np.random.seed(random_seed)
            np.random.shuffle(inds)
        return Split(inds[:train_n]), Split(inds[train_n:train_n+val_n]), Split(inds[train_n+val_n:train_n+val_n+test_n])
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return repr(self.data)
    
    def save(self, path):
        if path.split(".")[-1] == "npy":
            np.save(path, self.data)
        else:
            pickle.dump(self.data, open(path, 'wb'))

        
def load_data(path:str):
    if path.split(".")[-1] == "npy":
        return Data(np.load(path))
    return Data(pickle.load(open(path, 'rb')))
    
class Split:
    def __init__(self, inds:np.ndarray):
        self.inds = inds
    
    def __len__(self):
        return len(self.inds)
    
    def __repr__(self):
        return repr(self.inds)
    
    def save(self, path):
        pickle.dump(self.inds, open(path, 'wb'))

    def get_data(self, data:Data):
        return data.data[self.inds]
    
    def shuffle(self):
        np.random.shuffle(self.inds)


def load_split(path:str):
    return Split(pickle.load(open(path, 'rb')))


class Batches:
    def __init__(self, split:Split, batch_size:int):
        self.batch_size = batch_size
        self.split = split
        self.batch_N = int(math.ceil(len(split)/batch_size))
    
    def get_batch_inds(self, batch_i:int):
        assert(batch_i < self.batch_N)
        return self.split.inds[self.batch_size*batch_i:min(self.batch_size*(batch_i+1), len(self.split))]
    
    def get_data(self, data:Data, batch_i:int):
        return data.data[self.get_batch_inds(batch_i)]

def encode_y(y):
    n_classes = max([max(e) for e in y if len(e)])+1
    result = np.zeros((len(y), n_classes), dtype=int)
    for i,e in enumerate(y):
        for c in e:
            result[i,c] = 1
    return result


class Pack:
    def __init__(self, data, cuda=False):
        self.data = data
        self.pack_size = len(data)

        self.lengths = np.array([len(d) for d in data], dtype=int)
        self.max_length = int(max(self.lengths))

        self.ind_map_unsorted_to_sorted = np.argsort(self.lengths)[::-1]
        self.ind_map_sorted_to_unsorted = np.empty(self.pack_size, dtype=int)

        self.pack_lengths = np.zeros(self.pack_size, dtype=int)
        
        for k,v in enumerate(self.ind_map_unsorted_to_sorted):
            self.ind_map_sorted_to_unsorted[v] = k

        result = torch.zeros(self.pack_size, self.max_length).type(torch.LongTensor)

        for i in range(self.pack_size):
            sent = data[i]
            for word_i in range(len(sent)):
                result[self.ind_map_sorted_to_unsorted[i], word_i] = int(sent[word_i])
            self.pack_lengths[self.ind_map_sorted_to_unsorted[i]] = len(sent)
        
        self.pack_data = result
        if cuda:
            self.pack_data = self.pack_data.cuda()
        self.cuda = cuda
    
    def get_pack(self, embedding, torch_var=False):
        if not torch_var:
            emb = embedding(self.pack_data)
        else:
            emb = embedding(torch.autograd.Variable(self.pack_data))
        return torch.nn.utils.rnn.pack_padded_sequence(emb, self.pack_lengths, batch_first=True)

    def get_rev(self):
        if self.cuda:
            return torch.from_numpy(self.ind_map_sorted_to_unsorted).cuda()
        else:
            return torch.from_numpy(self.ind_map_sorted_to_unsorted)
    
    def get_lengths(self):
        return self.lengths
    
    def get_lengths_var(self):
        if self.cuda:
            return torch.autograd.Variable(torch.from_numpy(self.get_lengths()).type(torch.FloatTensor)).cuda()
        else:
            return torch.autograd.Variable(torch.from_numpy(self.get_lengths()).type(torch.FloatTensor))

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