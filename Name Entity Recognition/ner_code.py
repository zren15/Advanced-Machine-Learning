import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd


# if a word is not in your vocabulary use len(vocabulary) as the encoding
class NERDataset(Dataset):
    def __init__(self, df_enc):
        self.df = df_enc

        self.X = []
        self.Y = []
        for i in range(len(self.df)):
            if len(self.df[i:i+5]['word']) < 5:
                break
            else:
                x_near =  self.df[i:i+5]['word']
                y_near = self.df.iloc[i+2]['label'] # get the center of label
            
            self.X.append(x_near.values)
            self.Y.append(y_near)


    def __len__(self):
        """ Length of the dataset """
        ### BEGIN SOLUTION
        L = len(self.X)
        ### END SOLUTION
        return L

    def __getitem__(self, idx):
        """ returns x[idx], y[idx] for this dataset
        
        x[idx] should be a numpy array of shape (5,)
        """
        ### BEGIN SOLUTION
        x, y = self.X[idx],self.Y[idx]
        ### END SOLUTION
        return x, y 


def label_encoding(cat_arr):

   """ Given a numpy array of strings returns a dictionary with label encodings.

   First take the array of unique values and sort them (as strings). 
   """
   ### BEGIN SOLUTION
   cat_arr = cat_arr.astype(str)
   uniq_list = sorted(list(set(cat_arr)))
   vocab2index = {}
   for index, text in enumerate(uniq_list):
      vocab2index[text] = index
    ### END SOLUTION
   return vocab2index


def dataset_encoding(df, vocab2index, label2index):
    """Apply vocab2index to the word column and label2index to the label column

    Replace columns "word" and "label" with the corresponding encoding.
    If a word is not in the vocabulary give it the index V=(len(vocab2index))
    """
    V = len(vocab2index)
    df_enc = df.copy()
    ### BEGIN SOLUTION
    df_enc['word'] = np.array([vocab2index.get(word, V) for word in df_enc['word']])
    df_enc['label'] = np.array([label2index.get(label, V) for label in df_enc['label']])
    ### END SOLUTION
    return df_enc


class NERModel(nn.Module):
    def __init__(self, vocab_size, n_class, emb_size=50, seed=3):
        """Initialize an embedding layer and a linear layer
        """
        super(NERModel, self).__init__()
        torch.manual_seed(seed)
        ### BEGIN SOLUTION
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(5*emb_size, n_class)        
        ### END SOLUTION
        
    def forward(self, x):
        """Apply the model to x
        
        1. x is a (N,5). Lookup embeddings for x
        2. reshape the embeddings (or concatenate) such that x is N, 5*emb_size 
           .flatten works
        3. Apply a linear layer
        """
        ### BEGIN SOLUTION
        x = self.emb(x)
        x = x.flatten(1)
        x = self.linear(x)        
        ### END SOLUTION
        return x

def get_optimizer(model, lr = 0.01, wd = 0.0):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optim

def train_model(model, optimizer, train_dl, valid_dl, epochs=10):
    for i in range(epochs):
        ### BEGIN SOLUTION
        losses = []
        model.train()
        for x, y in train_dl:
            y_hat = model(x)
            loss = F.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        train_loss = np.mean(losses)      
        ### END SOLUTION
        valid_loss, valid_acc = valid_metrics(model, valid_dl)
        print("train loss  %.3f val loss %.3f and accuracy %.3f" % (
            train_loss, valid_loss, valid_acc))

def valid_metrics(model, valid_dl):
    ### BEGIN SOLUTION
    model.eval()
    losses = []
    acc = 0
    total = 0
    for x, y in valid_dl:
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        losses.append(loss.item())
        acc += (y_hat == y.detach().numpy()).sum()
        total += y.shape[0]  
    val_acc = acc/total  
    val_loss = np.mean(losses)
    ### END SOLUTION
    return val_loss, val_acc

