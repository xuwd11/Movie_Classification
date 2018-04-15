import torch
import torch.nn.functional as F
import numpy as np
from pack import Pack

class Encoder(torch.nn.Module):
    def __init__(self, encoder, embedding, hidden_dim, input_channel, num_layers, bidirectional, cuda):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.bidirectional = bidirectional
        self.cuda = cuda
            
        self.encoder = encoder(input_size=input_channel, hidden_size=hidden_dim, batch_first=True,
                                bidirectional=bidirectional, num_layers=num_layers)
        
        if cuda:
            self.embedding.cuda()
            self.encoder.cuda()

    def forward(self, pack:Pack):
        lengths = pack.get_lengths_var()
        rev = pack.get_rev()
        data = pack.get_pack(self.embedding)
        if self.cuda:
            lengths.cuda()
            rev.cuda()
        states_packed, _ = self.encoder(data) # (packed_sequence, hidden_state)
        states, _ = torch.nn.utils.rnn.pad_packed_sequence(states_packed)
        if self.bidirectional:
            states = states[:, :, :self.hidden_dim] + states[:, :, self.hidden_dim:]
        states_mean = torch.sum(states, dim=0) / lengths.unsqueeze(1)
        return states_mean[rev, :]
        
    def init_hidden(self):
        pass


class CombinedEncoder(torch.nn.Module):
    def __init__(self, encoder, embedding, hidden_dim, input_channel, num_layers, bidirectional, cuda):
        super().__init__()
        if isinstance(num_layers, int):
            nl1 = num_layers
            nl2 = num_layers
        else:
            nl1 = num_layers[0]
            nl2 = num_layers[1]
        self.title_encoder = Encoder(encoder=encoder, embedding=embedding, hidden_dim=hidden_dim, input_channel=input_channel, num_layers=nl1, bidirectional=bidirectional, cuda=cuda)
        self.text_encoder = Encoder(encoder=encoder, embedding=embedding, hidden_dim=hidden_dim, input_channel=input_channel, num_layers=nl2, bidirectional=bidirectional, cuda=cuda)

    def init_hidden(self):
        self.title_encoder.init_hidden()
        self.text_encoder.init_hidden()

    def forward(self, title_pack, text_pack):
        title_mean = self.title_encoder(title_pack)
        text_mean = self.text_encoder(text_pack)
        encoding = F.normalize((text_mean + title_mean) / 2, dim=1)
        return encoding
    
class MultiLayerFCReLUClassifier(torch.nn.Module):
    def __init__(self, dims, num_class, encoding_size, cuda):
        super().__init__()
        assert(len(dims)>0)
        self.fc1 = torch.nn.Linear(encoding_size, dims[0])
        self.relu1 = torch.nn.ReLU()
        if cuda:
            self.fc1.cuda()
            self.relu1.cuda()
        self.fcs = []
        self.relus = []
        prev_dim = dims[0]
        for dim in dims[1:]:
            fc = torch.nn.Linear(prev_dim, dim)
            relu = torch.nn.ReLU()
            if cuda:
                fc.cuda()
                relu.cuda()
            self.fcs.append(fc)
            self.relus.append(relu)
            prev_dim = dim
        
        self.out_fc = torch.nn.Linear(dims[-1], num_class)
        if cuda:
            self.out_fc.cuda()

    def forward(self, encodings):
        l_out = self.fc1(encodings)
        l_out = self.relu1(l_out)
        for i in range(len(self.fcs)):
            l_out = self.fcs[i](l_out)
            l_out = self.relus[i](l_out)
        out = self.out_fc(l_out)
        return out
    
class Model1(torch.nn.Module):
    def __init__(self, embedding, hidden_dim, num_layers, cuda):
        super(Model1, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = CombinedEncoder(encoder=torch.nn.GRU, embedding=embedding, hidden_dim=self.hidden_dim, input_channel=embedding.embedding_dim, num_layers=num_layers, bidirectional=True, cuda=cuda)
        self.classifier = MultiLayerFCReLUClassifier(dims=[256,128], num_class=19, encoding_size=self.hidden_dim, cuda=cuda)
    
    def forward(self, title_pack, text_pack):
        encodings = self.encoder(title_pack, text_pack)
        output = self.classifier(encodings)
        return output