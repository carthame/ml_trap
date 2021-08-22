import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

def make_data_set(data,lookback):

    X=[]
    Y=[]
    for i in range(len(data)-lookback):
        X.append([[data[j][0]] for j in range(i,i+lookback)])
        Y.append(data[i+lookback])

    
    return X,Y

def makeRandomBatch(X, Y, batch_size=10):

    batch_x = []
    batch_y = []

    for _ in range(batch_size):
        idx = np.random.randint(0, len(X) - 1)
        batch_x.append(X[idx])
        batch_y.append(Y[idx])
    
    return torch.tensor(batch_x).float(), torch.tensor(batch_y).float()

class LSTMNet(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(LSTMNet, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0)
        output = self.output_layer(output[:, -1, :])

        return output

