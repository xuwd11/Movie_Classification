import numpy as np
from itertools import chain
from pack import Pack
import torch

class Model:
    def predict(self, batches, batch_i):
        raise NotImplemented
    
    def get_y(self, batches, batch_i):
        raise NotImplemented
        
    def calculate_class(self, predictions):
        raise NotImplemented
    
    def train(self):
        raise NotImplemented
        
    def eval(self):
        raise NotImplemented
    
    def zero_grad(self):
        raise NotImplemented
    
    def parameters(self):
        raise NotImplemented


class EncoderClassifierModel(Model):
    def __init__(self, encoder, classifier):
        self.encoder = encoder
        self.classifier = classifier
    
    def train(self):
        self.encoder.train()
        self.classifier.train()
        return self
        
    def eval(self):
        self.encoder.eval()
        self.classifier.eval()
        return self
    
    def zero_grad(self):
        self.encoder.zero_grad()
        self.classifier.zero_grad()
    
    def parameters(self):
        return chain(self.encoder.parameters(), self.classifier.parameters())

        
class TextOnlyModel(EncoderClassifierModel):
    def __init__(self, encoder, classifier, texts, genres):
        super().__init__(encoder, classifier)
        self.texts = texts
        self.genres = genres
    
    def predict(self, batches, batch_i):
        texts = batches.get_data(self.texts, batch_i)
        text_pack = Pack(texts, cuda=True)
        return self.classifier(self.encoder(text_pack))
    
    def get_y(self, batches, batch_i):
        return batches.get_data(self.genres, batch_i)
        
    def calculate_class(self, predict_output):
        return predict_output>0
    

class PosterOnlyModel(EncoderClassifierModel):
    def __init__(self, encoder, classifier, posters, genres):
        super().__init__(encoder, classifier)
        self.posters = posters
        self.genres = genres
    
    def predict(self, batches, batch_i):
        images = batches.get_data(self.posters, batch_i)
        images = torch.autograd.Variable(torch.from_numpy(images)).cuda()
        return self.classifier(self.encoder(images).view(len(images),-1))
    
    def get_y(self, batches, batch_i):
        return batches.get_data(self.genres, batch_i)
        
    def calculate_class(self, predict_output):
        return predict_output>0

    
class TextPosterCombinedModel(EncoderClassifierModel):
    def __init__(self, encoder, classifier, texts, posters, genres):
        super().__init__(encoder, classifier)
        self.texts = texts
        self.posters = posters
        self.genres = genres
    
    def predict(self, batches, batch_i):
        texts = batches.get_data(self.texts, batch_i)
        text_pack = Pack(texts, cuda=True)
        
        images = batches.get_data(self.posters, batch_i)
        images = torch.autograd.Variable(torch.from_numpy(images)).cuda()
        
        encodings = self.encoder(text_pack, images)
        encodings = torch.cat(encodings, dim=1)
        
        return self.classifier(encodings)
    
    def get_y(self, batches, batch_i):
        return batches.get_data(self.genres, batch_i)
        
    def calculate_class(self, predictions):
        return predictions>0