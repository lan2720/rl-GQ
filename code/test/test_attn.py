from PointerNet import Attention
import torch

src_dim=5
tgt_dim=7
attn_dim=6
batch_size = 3
seq_len = 5

attn = Attention('general', src_dim, tgt_dim, attn_dim)

ht = torch.rand(batch_size, tgt_dim)
context = torch.rand(batch_size, seq_len, src_dim)
context[0, 3:, :] = 0.0
context[1, 4:, :] = 0.0

attn_context, alpha = attn(ht, context)
print('attn_context:', attn_context)
print('alpha:', alpha)
