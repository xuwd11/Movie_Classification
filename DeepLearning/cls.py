import torch
from torch_models import MultiLayerFCReLUClassifier

class BRClassifier(torch.nn.Module):
    def __init__(self, dims, num_class, encoding_size, cuda):
        super().__init__()
        
        self.classifiers = []
        for i in range(num_class):
            cls = MultiLayerFCReLUClassifier(dims, 1, encoding_size, cuda)
            self.add_module(str(i), cls)
            self.classifiers.append(cls)

    def forward(self, encodings):
        out = torch.stack([cls(encodings) for cls in self.classifiers])[:,:,0]
        return torch.transpose(out,0,1)