import numpy as np
import math
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
        pickle.dump(self.data, open(path, 'wb'))

        
def load_data(path:str):
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