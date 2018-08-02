import numpy as np
import torch
import torch.nn as nn
from PointerNet import length_and_mask, restore_order

vocab_size = 10
embed_dim = 5
hid_dim = 6
inputs = np.array([[3,6,2,0,0,0,0], [4,1,2,9,3,0,0], [4,7,1,3,0,0,0], [2,0,0,0,0,0,0]])
inputs_var = torch.from_numpy(inputs).long()
inputs_length, inputs_mask = length_and_mask(inputs_var)

embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
lstm  = nn.LSTM(input_size=embed_dim, hidden_size=hid_dim, batch_first=True, bidirectional=True)

embedded_inputs = embedding(inputs_var)

# sort input
sorted_inputs_length, sorted_indices = torch.sort(inputs_length, descending=True)
sorted_embedded_inputs = embedded_inputs[sorted_indices]

batch_size, seq_len, _ = embedded_inputs.size()
embedded_inputs = nn.utils.rnn.pack_padded_sequence(sorted_embedded_inputs, sorted_inputs_length, batch_first=True)
sorted_outputs, sorted_hidden = lstm(embedded_inputs)
sorted_outputs, _ = nn.utils.rnn.pad_packed_sequence(sorted_outputs, batch_first=True)
#_, raw_indices = torch.sort(sorted_indices)
#outputs= sorted_outputs[raw_indices]#, sorted_hidden[raw_indices]
outputs = restore_order(sorted_outputs, sorted_indices)


#if outputs.size(1) < seq_len:
#    hid_dim = outputs.size(2) # if bidir, hid_dim = 2*self.hidden_dim else self.hidden_dim
#    dummy_tensor = torch.tensor(torch.zeros(batch_size, seq_len-outputs.size(1), hid_dim))
#    outputs = torch.cat([outputs, dummy_tensor], 1)

print(outputs.size())
print(outputs)
        

