import torch
import torch.nn.functional as F
from pack import Pack

class TextPosterCombinedEncoder(torch.nn.Module):
    def __init__(self, text_encoder, poster_encoder):
        super().__init__()
        self.text_encoder = text_encoder
        self.poster_encoder = poster_encoder
        
    def forward(self, text_pack, posters):
        poster_encodings = self.poster_encoder(posters).view(len(posters),-1)
        text_encodings = self.text_encoder(text_pack)
        return poster_encodings, text_encodings

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
    
    
class Encoder2(torch.nn.Module):
    def __init__(self, encoder, embedding, hidden_dim, input_channel, num_layers, bidirectional, dropout, cuda):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.bidirectional = bidirectional
        self.cuda = cuda

        self.encoder = encoder(input_size=input_channel, hidden_size=hidden_dim, batch_first=True,
                                bidirectional=bidirectional, num_layers=num_layers, dropout=dropout)
        
        if cuda:
            self.embedding.cuda()
            self.encoder.cuda()

    def forward(self, pack:Pack):
        rev = pack.get_rev()
        data = pack.get_pack(self.embedding, torch_var=True)
        if self.cuda:
            rev.cuda()
        states_packed, _ = self.encoder(data) # (packed_sequence, hidden_state)
        states, _ = torch.nn.utils.rnn.pad_packed_sequence(states_packed)
        states = torch.cat([states[-1,:,:self.hidden_dim], states[0,:,self.hidden_dim:]], dim=1)
        return states[rev, :]
        
    def init_hidden(self):
        pass

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