import torch
import torch.nn as nn
import numpy as np

sentences = [[2,5,7,1,3,6,3,5,4,0], [6,3,1, 3,5,3,0,0,0, 0]]
ans_start_end = [(3,5), (1,4)]

data = np.array(sentences)
ans_pos_indicator = np.zeros_like(data)
for i, (start, end) in enumerate(ans_start_end):
    ans_pos_indicator[i, start:end] = 1

print(ans_pos_indicator)


vocab_size = 10
embed_dim = 2
embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
data_tensor = torch.from_numpy(data)
embed_data = embedding(data_tensor)
ans_pos_tensor = torch.from_numpy(ans_pos_indicator).float().unsqueeze(-1)
augumented_data = torch.cat([embed_data, ans_pos_tensor], -1)
print(augumented_data)
