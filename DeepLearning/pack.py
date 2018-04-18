import numpy as np

import torch

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
    
    def get_pack(self, embedding):
        emb = embedding(self.pack_data)
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

