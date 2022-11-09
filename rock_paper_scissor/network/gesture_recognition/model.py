import torch.nn as nn


class Hand_MLP(nn.Module):
    """MLP encoder module."""
    def __init__(self, n_in, n_hid, n_out):
        super(Hand_MLP, self).__init__()
        self.dropout_1 = nn.Dropout(p = 0.2)
        self.fc1 = nn.Linear(n_in, n_hid, bias = True)
        self.dropout_2 = nn.Dropout(p = 0.4)
        self.fc2 = nn.Linear(n_hid, 10, bias = True)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(10,n_out, bias = True)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.dropout_1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
