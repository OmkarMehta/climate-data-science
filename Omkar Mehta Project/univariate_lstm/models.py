import torch
import torch.nn as nn

import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        Deep Neural Network 
        '''
        super(DNN, self).__init__()

        # Define the network: a sequence of layers
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size), # input layer
            nn.ReLU(inplace=True), # activation function
            nn.Linear(hidden_size, output_size) # output layer
        )

    def forward(self, x):
        # Forward pass through the network
        x = x.squeeze(dim=2) # remove the last dimension because the input is 3-dimensional for LSTM
        out = self.main(x) # pass the input to the network
        return out # return the output


class CNN(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        '''
        Convolutional Neural Network
        '''
        super(CNN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_dim, kernel_size=1), # input layer
            nn.ReLU(), # activation function

            nn.Flatten(), # flatten the output of the convolutional layer

            nn.Linear(hidden_dim, 10), # fully connected layer
            nn.Linear(10, output_size) # output layer
        )

    def forward(self, x):
        out = self.main(x)
        return out


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        '''
        Recurrent Neural Network (Vanilla RNN), inspired by the RNN paper 
        '''
        super(RNN, self).__init__()

        self.input_size = input_size # input size
        self.hidden_size = hidden_size # hidden size
        self.num_layers = num_layers # number of layers
        self.output_size = output_size # output size

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True) # batch_first=True: input & output will have shape (batch_size, seq_len, input_size)

        self.fc = nn.Linear(hidden_size, output_size) # output layer

    def forward(self, x):
        out, _ = self.rnn(x) # out: tensor of shape (batch_size, seq_len, hidden_size), _: hidden state
        out = out[:, -1, :] # take the last output of the RNN
        out = self.fc(out) # pass the output of the RNN to the output layer

        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        """
        Long Short Term Memory (LSTM)
        """
        super(LSTM, self).__init__()

        self.input_size = input_size # input size
        self.hidden_size = hidden_size # hidden size
        self.num_layers = num_layers # number of layers
        self.output_size = output_size # output size
        self.bidirectional = bidirectional # bidirectional LSTM: True or False (default) if True, the LSTM will have 2*hidden_size output size

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional) # batch_first=True: input & output will have shape (batch_size, seq_len, input_size)

        if self.bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size) # output layer if bidirectional
        else:
            self.fc = nn.Linear(hidden_size, output_size) # output layer if not bidirectional

    def forward(self, x):
        out, _ = self.lstm(x) # out: tensor of shape (batch_size, seq_len, hidden_size), _: hidden state
        out = out[:, -1, :] # take the last output of the LSTM
        out = self.fc(out) # pass the output of the LSTM to the output layer
        # return out
        return out

class AttentionalLSTM(nn.Module):
    
    def __init__(self, input_size, qkv, hidden_size, num_layers, output_size, bidirectional=False):
        """
        LSTM with Attention
        """
        super(AttentionalLSTM, self).__init__()

        self.input_size = input_size # input size
        self.qkv = qkv # number of queries, keys and values (qkv)
        # in attentional LSTM, the number of queries, keys and values (qkv) are equal
        self.hidden_size = hidden_size # hidden size
        self.num_layers = num_layers # number of layers
        self.output_size = output_size # output size

        self.query = nn.Linear(input_size, qkv) # query layer
        self.key = nn.Linear(input_size, qkv) # key layer
        self.value = nn.Linear(input_size, qkv) # value layer

        self.attn = nn.Linear(qkv, input_size) # attention layer
        self.scale = math.sqrt(qkv) # scale factor

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional) # batch_first=True: input & output will have shape (batch_size, seq_len, input_size)

        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size) # output layer if bidirectional
        else:
            self.fc = nn.Linear(hidden_size, output_size) # output layer if not bidirectional

    def forward(self, x):

        Q, K, V = self.query(x), self.key(x), self.value(x) # Q, K, V: tensor of shape (batch_size, seq_len, qkv)

        dot_product = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale # dot product: tensor of shape (batch_size, seq_len, seq_len)
        scores = torch.softmax(dot_product, dim=-1) # softmax: tensor of shape (batch_size, seq_len, seq_len)
        scaled_x = torch.matmul(scores, V) + x # scaled_x: tensor of shape (batch_size, seq_len, qkv)

        out = self.attn(scaled_x) + x # out: tensor of shape (batch_size, seq_len, qkv)
        out, _ = self.lstm(out) # out: tensor of shape (batch_size, seq_len, hidden_size), _: hidden state
        out = out[:, -1, :] # take the last output of the LSTM
        out = self.fc(out) # pass the output of the LSTM to the output layer
        # return out
        return out