

vocab_dists = 

def calc_final_dist(vocab_dists, attn_dists):
    batch_size = 2
    extended_size = 10
    extended_vsize = self.vocab_size + self.max_para_oovs
    attn_dists_projected =  torch.zeros(self.hps.batch_size, extended_vsize).scatter_add_(1, self.paragraph_inputs_extend_vocab, copy_dist)
    
    final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]
    return final_dists


