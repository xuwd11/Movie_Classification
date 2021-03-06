import sys
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from pack import Pack
from common import Split, Batches
import torch
import torch.nn.functional as F
import numpy as np
from model import Model

class ProgressBar:
    def __init__(self, total, display=True):
        self.cur = 0
        self.total = total
        self.prev = 0
        self.display = display
    
    def print(self, txt, end=None):
        if not self.display:
            return
        if end is None:
            print(txt)
        else:
            print(txt, end=end)
    
    def tick(self):        
        self.cur += 1
        cur_p = self.cur / self.total
        self.print(".", end="")
        if cur_p * 100 >= self.prev + 5:
            self.prev += 5
            self.print(self.prev, end="")
            if self.prev >= 100:
                self.print(" F")
        sys.stdout.flush()
    
    def reset(self):
        self.cur = 0
        self.prev = 0
        
def evaluate(split:Split, model:Model, batch_size:int):
    p,t=inference(split=split, model=model, batch_size=batch_size)
    pr = precision_score(p,t,average='micro')
    rc = recall_score(p,t,average='micro')
    f1 = f1_score(p,t,average='micro')
    
    print("P(mi)", pr, "\tR(mi):", rc, "\tF1(mi):", f1)
    print()
    
    return pr, rc, f1

def inference(split:Split, model:Model, batch_size:int, proba=False):
    batches = Batches(split, batch_size)

    model = model.eval()
    
    preds = []
    trues = []
    
    for i in range(batches.batch_N):
        
        model_output = model.predict(batches, i)
        
        y_pred = model_output.cpu().data.numpy()

        if not proba:
            y_pred = model.calculate_class(y_pred)
            
        y_true = model.get_y(batches, i)
        
        preds.append(y_pred)
        trues.append(y_true)

    model = model.train()
    
    return np.concatenate(preds), np.concatenate(trues)