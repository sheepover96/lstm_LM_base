import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMLM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_layers, hidden_dim, gpu):
        super(LSTMLM, self).__init__()
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.continue_seq = True
        self.return_seq = False

        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        h0, c0 = (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
                  weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))
        if self.gpu:
            h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0
        #self.h0, self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim), torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        #if self.gpu:
        #    self.h0, self.c0 = self.h0.cuda(), self.c0.cuda()
    
    def set_embed_parameter(self, weights):
        self.encoder.weights = nn.Parameter(weights)
        self.encoder.weights.requires_grad = False

    def forward(self, x, hidden):
        emb = self.encoder(x)
        output, hidden = self.lstm(emb, hidden)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
        #if self.continue_seq:
        #    h, (self.h0, self.c0) = self.lstm(x, (self.h0, self.c0))
        #else:
        #    self.reset_state(x)
        #    h, _ = self.lstm(x, (self.h0, self.c0))

        #if self.return_seq:
        #    pass
        #else:
        #    h =  h[-1,:,:]

        #return h
        #lstm_out, self.hidden = self.lstm(
        #    embeds.view(len(x), 1, -1), self.hidden)