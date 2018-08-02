# coding:utf-8
from PointerNet import Encoder
import torch.nn.functional as F
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

#x = [[2,4,6], [3,5,7,4,2], [2,1]]
sentences = [[2,5,7,1,3,6,3,5,4], [6,3,1, 3,5,3]]
ans_start_end = [(3,5), (1,4)]

data = np.zeros((2, 10), dtype=int)
for i, xi in enumerate(sentences):
    data[i, :len(xi)] = xi
data_tensor = torch.from_numpy(data)
print('numpy data:', data)
print('data tensor:', data_tensor)

embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

encoder = Encoder(embed_dim, hid_dim, n_layers, dropout=0, bidir=bidir)

data_len, data_mask = length_and_mask(data_tensor)
sorted_data_len, sorted_data_indices = torch.sort(data_len, descending=True)
print('sorted data length:', sorted_data_len, 'sorted data indices:', sorted_data_indices)
sorted_data_tensor = data_tensor[sorted_data_indices]
print('sorted data tensor:', sorted_data_tensor)

embedded_sorted_data = embedding(sorted_data_tensor)
print('embeded sort data tensor:', embedded_sorted_data)
outputs, ht = encoder(embedding(sorted_data_tensor), sorted_data_len, encoder.init_hidden(sorted_data_tensor))
print('outputs:', outputs)
print('output size:', outputs.size())

# 根据outputs=[batch_size, seq_len, hid_dim](超出真实长度的部分全为0 vector)，得到[batch_size, seq_len]的mask
source_mask = torch.eq(torch.mean(outputs, -1), 0)
print('source_mask:', source_mask)

# 在outputs中将ans pos位置的ht拿出来，作为h^a
# outputs = [batch_size, seq_len, hid_dim*n_dir]
# [batch_size, max_ans_len, hid_dim*n_dir] -> lstm final state -> [1 * num_directions, batch_size, hid_dim]
# NOTE: n_layers for ans_encoder must be 1
ans_encoder = Encoder(outputs.size(2)+embed_dim, hid_dim, n_layers=1, dropout=0, bidir=bidir)
ans_length = [(end-start) for start, end in ans_start_end]
# 取出h_j^d
ans_hidden = torch.zeros_like(outputs)
for i, (start, end) in enumerate(ans_start_end):
    ans_hidden[i,0:ans_length[i]] = outputs[i,start:end]
# 取出a_j
embedded_ans = torch.zeros_like(embedded_sorted_data)
for i, (start, end) in enumerate(ans_start_end):
    embedded_ans[i, 0:ans_length[i]] = embedded_sorted_data[i, start:end]
comb_h = torch.cat((ans_hidden, embedded_ans), dim=-1)
# 根据ans length排序, 排序后才能进行lstm
sorted_ans_len, sorted_ans_indices = torch.sort(torch.tensor(ans_length), descending=True)
sorted_comb_h = comb_h[sorted_ans_indices]
print('sort comb_h:', sorted_comb_h)
print('sort ans len:', sorted_ans_len)
_, (ha, _) = ans_encoder(sorted_comb_h, sorted_ans_len, ans_encoder.init_hidden(sorted_comb_h))
print('before fix shape:', ha)
ha = ans_encoder.fix_hidden(ha).squeeze(0)
#ca = ans_encoder.fix_hidden(ca).squeeze(0)
# ha  = (num_layers * num_directions, batch, hidden_size) = [1*2, batch_size, hid_dim]
print('after fix shape:', ha)
print('ha:', ha.size())
#print('ca:', ca.size())
print('embedded answer:', embedded_ans)
print('ans pos:', ans_start_end)
#print('ans_hidden:', ans_hidden)

h = hid_dim*2 if bidir else hid_dim
L = nn.Parameter(torch.FloatTensor(h, h), requires_grad=True) 
W0 = nn.Parameter(torch.FloatTensor(h, hid_dim), requires_grad=True) 
b0 = nn.Parameter(torch.FloatTensor(hid_dim), requires_grad=True) 
# initialize weights
nn.init.uniform_(L, -1, 1)
nn.init.uniform_(W0, -1, 1)
nn.init.uniform_(b0, -1, 1)
def combine_hidden(answer_hidden, paragraph_hidden):
    """
    refer to the paper of `` in Sec 3.1, the first function
    @params:
        - encode_answer: ha or ca, each of shape = [batch_size, hid_dim*n_dir]
        - encode_paragraph: [batch_size, seq_len, hid_dim*n_dir]
    """
    global L, W0, b0
    #batch_size, hidden_dim = answer_hidden.size()
    #print(answer_hidden.size())
    #print(paragraph_hidden.size())
    #print(torch.mean(paragraph_hidden, 1).size())
    r = torch.mm(answer_hidden, L) + torch.mean(paragraph_hidden, 1)
    #print(r.size())
    return F.tanh(torch.mm(r, W0) + b0)
    

h0 = combine_hidden(ha, outputs)
print('h0 for decoder:', h0)
print('h0 size:', h0.size())


#print(ht[0])
#print(ht[0].size())
#print(ht[1])
#linear = nn.Linear(hid_dim*2 if bidir else hid_dim, 1)

#batch_size, seq_len, _ = outputs.size()
#haha = linear(outputs.reshape(batch_size*seq_len, -1)).reshape(batch_size, seq_len, -1)
#print(haha)
