import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embedding, 
                 hidden_dim,
                 n_layers, bidir,
                 dropout_p=0.0, input_dropout_p=0.0):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers*2 if bidir else n_layers
        self.bidir = bidir
        self.embedding = embedding
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.lstm = nn.LSTM(self.embedding.embedding_dim,
                            hidden_dim,
                            n_layers,
                            batch_first=True,
                            dropout=dropout_p,
                            bidirectional=bidir)

    def forward(self, input_var, input_length):
        batch_size, seq_len = input_var.size()
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_length, batch_first=True)
        outputs, hidden = self.lstm(embedded)
        
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        if outputs.size(1) < seq_len:
            hid_dim = outputs.size(2)
            dummy_tensor = torch.zeros(batch_size, seq_len-outputs.size(1), hid_dim, 
                                       device=torch.device('cuda'))
            outputs = torch.cat([outputs, dummy_tensor], 1)
        return outputs, hidden
        # outputs shape = [batch_size, seq_len, hid_size * n_dir]
        # hidden shape = [n_layer * n_dir, batch_size, hid_size]
