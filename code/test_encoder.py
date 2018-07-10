from PointerNet import Encoder
import torch.nn as nn
import torch
import numpy as np

def length_and_mask(input_tensor):
    mask = torch.eq(input_tensor, 0)
    length = input_tensor.size(1) - torch.sum(mask, dim=1)
    return length, mask


def masked_softmax(inputs, mask):
    """
    @params:
        - inputs: [key_dim, val_dim]
        - mask: []
    """
    pass


vocab_size = 10
embed_dim = 2
hid_dim = 2
n_layers = 1
bidir = True

x = [[2,4,6], [3,5,7,4,2], [2,1]]
data = np.zeros((3, 5), dtype=int)
for i, xi in enumerate(x):
    data[i, :len(xi)] = xi

data_tensor = torch.from_numpy(data)


print(data)
print(data_tensor)
embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

#embedded_data = embedding(data_tensor)
#print(embedded_data)

encoder = Encoder(embed_dim, hid_dim, n_layers, dropout=0, bidir=bidir)
data_len, data_mask = length_and_mask(data_tensor)
sorted_data_len, sorted_data_indices = torch.sort(data_len, descending=True)
print(sorted_data_len, sorted_data_indices)
sorted_data_tensor = data_tensor[sorted_data_indices]
print(sorted_data_tensor)

outputs, ht = encoder(embedding(sorted_data_tensor), sorted_data_len, encoder.init_hidden(sorted_data_tensor))
print(outputs)
print(outputs.size())

print(ht[0])
print(ht[0].size())
print(ht[1])
#linear = nn.Linear(hid_dim*2 if bidir else hid_dim, 1)

#batch_size, seq_len, _ = outputs.size()
#haha = linear(outputs.reshape(batch_size*seq_len, -1)).reshape(batch_size, seq_len, -1)
#print(haha)
