# coding:utf-8
import torch

#def forward_step(self, input_var, hidden, encoder_outputs, encoder_mask, function):
#    # input_var: dec inputs = [batch_size, dec_seq_len]
#    # hidden: dec initial h0
#    # encoder_outputs: [batch_size, enc_seq_len, hid]
#    # encoder_mask [batch_size, enc_seq_len]
#
#    # How to compute p_gen
#    # p_gen: a scalar  = sigmoid(wh*ht + ws*st + wx*xt + bias)
#    # ht: context vector (mix), st: decoder state, xt: decoder input
#    # ht = [batch_size, dec_seq_len, hid]
#    # st = [batch_size, dec_seq_len, hid]
#    # xt = [batch_size, dec_seq_len, emb_dim]
#    # -> p_gen = [batch_size, dec_seq_len]
#    # cat_fea = torch.cat([mix, dec_output, dec_input], dim=2)
#    # nn.Linear(hid*2+emb_dim, 1)
#    # linear(cat_fea) -> [batch_Size, dec_seq_len, 1] -> squeeze(2) -> [batch_size, dec_seq_len] as p_gen
#
#    
#    batch_size = input_var.size(0)
#    dec_len = input_var.size(1)
#    embedded = self.embedding(input_var)
#    embedded = self.input_dropout(embedded)
#    
#    output, hidden = self.lstm(embedded, hidden)
#
#    attn = None
#    if self.use_attention:
#        output, attn_dist = self.attention(output, encoder_outputs, encoder_mask)
#    if self.use_copy:
#        vocab_dist = F.softmax(self.out.contiguous().view(-1, self.hidden_dim), dim=1).view(batch_size, dec_len, -1)
#        predicted_softmax = calc_final_dist(p_gen, vocab_dist, attn_dist)
#        predicted_softmax = F.
#    predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_dim)), dim=1).view(batch_size, dec_len, -1)
#    return predicted_softmax, hidden, attn



def loss(target, vocab_dist, target_mask, use_copy=False,
    attn_dist=None, p_gen=None, enc_input_extended_vocab=None, max_enc_oovs=None):
    """
    target: [batch_size, dec_seq_len]
    vocab_dist: [batch_size, dec_seq_len, vocab_size]
    target_mask: [batch_size, dec_seq_len]
    attn_dist: [batch_size, dec_seq_len, enc_seq_len]
    p_gen = [batch_size, dec_seq_len]
    enc_input_extended_vocab: [batch_size, dec_seq_len]
    max_enc_oovs: a scalar

    不用attention不用copy
    loss = 1. 得到LogSoftmax，再经过NLL 2. 仅得到score，直接crossEntropy
        3. 得到softmax，再根据对应位置选出target的prob，再进行-log的操作
    
    使用attention，不使用copy
    loss = 同上
    
    使用attention，使用copy
    这时，softmax要combine vocab_dist和attn_dist的概率和，得到final_dist，这一步需要使用softmax后的结果
    也就是上述的3

    target = [batch_size, dec_seq_len]
    vocab_dist = [batch_size, dec_seq_len, vocab_size]
    """
    batch_size, dec_seq_len = target.size()
    vocab_size = vocab_dist.size(2)
    if not use_copy:
        # 都不适用 or 单纯使用attention没有copy
        # 1. 有vocab_dist和target, 但不用attention和copy
        # 将target部位的prob拿出来，得到[batch_size, dec_seq_len]
        final_dist = vocab_dist
    else:
        # 使用copy
        assert attn_dist is not None, "If use copy mechanism, please give attention distribution"
        assert p_gen is not None, "If use copy mechanism, please give p_gen (p_copy=1-p_gen)"
        assert enc_input_extended_vocab is not None, "If use copy mechanism, please give enc_input_extended_vocab"
        assert max_enc_oovs, "If use copy mechanism, please give max_enc_oovs, which is a scalar for each batch"
        # p_gen = [batch_size, dec_seq_len]
        p_gen.unsqueeze_(2)
        vocab_dist = p_gen * vocab_dist
        attn_dist = (1.0-p_gen) * attn_dist
        if max_enc_oovs > 0:
            extra_zeros = torch.zeros((batch_size, dec_seq_len, max_enc_oovs))
            vocab_dist_extended = torch.cat([vocab_dist, extra_zeros], dim=2)
        else:
            vocab_dist_extended = vocab_dist
        extended_vsize = vocab_size + max_enc_oovs
        # 由于enc_inputs_extend_vocab = [batch_size, enc_seq_len] 是二维的，scatter_add_这个的三个参数的维度都必须相同
        # 但attn_dist = [batch_size, dec_seq_len, enc_seq_len] 是三维，因此这里需要把attn_dist在dim=1 split开

        # 对于copy机制而言，enc_input中不存在任何unk了，因为所有的unk都会被新增的idx替代，构成一个oovs
        # target和dec_input中，如果遇到oovs中的词，会被替换成对应idx，但是仍有一些词会是unk
        attn_dist_by_step = [dist.squeeze(1) for dist in torch.split(attn_dist, 1, dim=1)]
        attn_dist_projected = [torch.zeros(batch_size, extended_vsize).scatter_add_(1, enc_input_extended_vocab, attn_dist) for attn_dist in attn_dist_by_step]
        attn_dist = torch.stack(attn_dist_projected, dim=1)
        final_dist = vocab_dist_extended + attn_dist # [batch_size, dec_seq_len, extend_vsize]
    target_prob = torch.gather(final_dist, dim=2, index=target.unsqueeze(2)).squeeze(2) # [batch_size, dec_seq_len]
    loss = -torch.log(target_prob) # [batch_size, dec_seq_len]
    # mask
    loss.masked_fill_(target_mask, 0.)
    assert loss.size(0) == batch_size and loss.size(1) == dec_seq_len
    return loss


def main():
    enc_seq_len = 5
    dec_seq_len = 5
    batch_size = 2
    vocab_size = 7
    attn_dist = torch.rand(batch_size, dec_seq_len, enc_seq_len)
    vocab_dist = torch.rand(batch_size, dec_seq_len, vocab_size) 
    target = torch.tensor([[2,5,7,1,4], [4,12,2,1,6]])
    target_mask = torch.tensor([[0,0,0,0,1], [0,0,0,0,1]], dtype=torch.uint8)
    max_enc_oovs = 10
    p_gen = torch.rand(batch_size, dec_seq_len) 
    enc_input_extended_vocab = torch.tensor([[2,5,1,7,11], [12,6,5,8,3]])
            
    a = loss(target, vocab_dist, target_mask, use_copy=True,
        attn_dist=attn_dist, p_gen=p_gen, enc_input_extended_vocab=enc_input_extended_vocab, max_enc_oovs=max_enc_oovs)
    print(a)

if __name__ == '__main__':
    main()
#def calc_final_dist(self, p_gen, vocab_dists, attn_dists):
#    vocab_dists = [p_gen * dist for (p_gen,dist) in zip(self.p_gens, vocab_dists)]
#    attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(self.p_gens, attn_dists)]
#    batch_size = vocab_dists[0].size(0)
#    if self.max_para_oovs > 0:
#        extra_zeros = torch.zeros((batch_size, self.max_para_oovs)).cuda()
#        vocab_dists_extended = [torch.cat([dist, extra_zeros], dim=1) for dist in vocab_dists]
#    else:
#        vocab_dists_extended = vocab_dists
#    extended_vsize = self.vocab_size + self.max_para_oovs
#    attn_dists_projected =  [torch.zeros(batch_size, extended_vsize).cuda().scatter_add_(1, self.paragraph_inputs_extend_vocab, copy_dist) for copy_dist in attn_dists]
#    
#    final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]
#    return final_dists
