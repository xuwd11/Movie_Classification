import torch
from model import Model
from utils import ProgressBar, evaluate
from common import Batches, Split
import numpy as np

def train_batches(model:Model, train:Split, loss, batch_size:int, optimizer, display:bool=True):
    losses = []

    batches = Batches(train, batch_size)

    pb = ProgressBar(batches.batch_N, display=display)
    
    train.shuffle()
    pb.reset()
    for i in range(batches.batch_N):
        model.zero_grad()
        optimizer.zero_grad()

        y_true = model.get_y(batches, i)
        y_true = torch.autograd.Variable(torch.from_numpy(y_true)).cuda().type(torch.cuda.FloatTensor)
        model_output = model.predict(batches, i)

        l = loss(model_output, y_true)
        l.backward()

        optimizer.step()

        losses.append(l.data.cpu().numpy()[0])

        pb.tick()
        
    return losses

def train_epoches(n_epochs:int, model:Model, train:Split, val:Split, loss, batch_size:int, optimizer, scheduler=None):
    epoch_losses = []
    for epoch in range(n_epochs):
        losses = train_batches(model=model, train=train,
                               loss=loss, batch_size=batch_size, optimizer=optimizer, display=True)
        epoch_losses.append(losses)
        print("epoch {}:".format(epoch), np.mean(losses))
        
        #print("Train:", end="\t")
        #train_result = evaluate(train, model, batch_size=batch_size)
        train_result = None
        
        print("Val:", end="\t")
        val_result = evaluate(val, model, batch_size=batch_size)
        
        if scheduler is not None:
            scheduler.step(val_result[-1])
    return epoch_losses, (train_result, val_result)