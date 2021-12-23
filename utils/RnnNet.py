import torch
from torch import nn

class RnnNet(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers):
        super(RnnNet, self).__init__()
        self.rnn_layer = nn.LSTM(
            num_inputs,
            num_hiddens,
            num_layers = num_layers
        )

        self.LinearSeq = nn.Sequential(
            nn.Linear(num_hiddens, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, X):
        X, _ = self.rnn_layer(X)
        # print('X.shape: {}'.format(X.shape))
        # print('hidden.shape: {}'.format(hidden.shape))
        return self.LinearSeq(X[:, -1, :]).reshape(-1)

if __name__ == '__main__':
    X = torch.rand((32, 120, 6))
    Net = RnnNet(6, 256, 2)
    print(Net(X).shape)