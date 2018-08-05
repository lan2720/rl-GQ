# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import sys

class Encoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        Initiate Encoder

        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers*2 if bidir else n_layers
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            n_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidir)

        self.h0 = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs, inputs_length, hidden):
        """
        Encoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor init_hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """
        batch_size, seq_len, _ = embedded_inputs.size()
        embedded_inputs = nn.utils.rnn.pack_padded_sequence(embedded_inputs, inputs_length, batch_first=True)
        outputs, hidden = self.lstm(embedded_inputs, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        if outputs.size(1) < seq_len:
            hid_dim = outputs.size(2) # if bidir, hid_dim = 2*self.hidden_dim else self.hidden_dim
            dummy_tensor = torch.tensor(torch.zeros(batch_size, seq_len-outputs.size(1), hid_dim)).cuda()
            outputs = torch.cat([outputs, dummy_tensor], 1)
        # outputs shape = [batch_size, seq_len, hid_size * n_dir]
        # hidden shape = [n_layer * n_dir, batch_size, hid_size]
        return outputs, hidden

    def initialize_hidden(self, embedded_inputs):
        """
        Initiate hidden units

        :param Tensor embedded_inputs: The embedded input of Pointer-NEt
        :return: Initiated hidden units for the LSTMs (h, c)
        """
        batch_size = embedded_inputs.size(0)
        # Reshaping (Expanding)
        # (num_layers * num_directions, batch, hidden_size)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.c0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        return h0, c0

    def fix_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.bidir:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h


class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, method, src_input_dim, tgt_input_dim, attn_dim):
        """
        Initiate Attention

        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()
        self.method = method
        
        self.Wa, self.Wh, self.Ws, self.va, self.b_attn = None, None, None, None, None
        if method == 'dot':
            assert src_input_dim == tgt_input_dim, 'Please make sure src_input_dim = tgt_input_dim when Attn=dot'
            self.score = self._dot
        elif method == 'general':
            self.Wa = nn.Parameter(torch.FloatTensor(tgt_input_dim, src_input_dim))
            self.score = self._general
        elif method == 'concat':
            self.Wh = nn.Parameter(torch.FloatTensor(tgt_input_dim, attn_dim))
            self.Ws = nn.Parameter(torch.FloatTensor(src_input_dim, attn_dim))
            self.b_attn = nn.Parameter(torch.FloatTensor(attn_dim,))
            self.va = nn.Parameter(torch.FloatTensor(attn_dim))
            self.score = self._concat

        self.softmax = nn.Softmax(dim=1)

        # Initialize weight
        if self.Wa is not None:
            nn.init.uniform_(self.Wa, -0.1, 0.1)
        if self.Wh is not None:
            nn.init.uniform_(self.Wh, -0.1, 0.1)
        if self.Ws is not None:
            nn.init.uniform_(self.Ws, -0.1, 0.1)
        if self.b_attn is not None:
            nn.init.uniform_(self.b_attn, -0.1, 0.1)
        if self.va is not None:
            nn.init.uniform_(self.va, -0.1, 0.1)
    
    def _dot(self, ht, context):
        # [batch_size, enc_seq_len]
        return torch.bmm(context, ht.unsqueeze(2)).squeeze(2)
    
    def _general(self, ht, context):
        return torch.bmm(context, torch.mm(ht, self.Wa).unsqueeze(2)).squeeze(2)

    def _concat(self, ht, context):
        batch_size, seq_len, _ = context.size()
        concat_tensor = torch.bmm(ht.unsqueeze(1).expand(-1, seq_len, -1), self.Wh.unsqueeze(0).expand(batch_size, -1, -1)) \
                      + torch.bmm(context, self.Ws.unsqueeze(0).expand(batch_size, -1, -1)) \
                      + self.b_attn
        # va = [attn_dim,] -> [1, attn_dim] -> [B, attn_dim] -> [B, attn, 1]
        return torch.bmm(F.tanh(concat_tensor), 
                self.va.unsqueeze(0).expand(batch_size, -1).unsqueeze(2)).squeeze(2)

    def forward(self, input, context):
        """
        Attention - Forward-pass

        :param Tensor input: Hidden state h, [batch_size, dec_hid_dim]
        :param Tensor context: Attention context, [batch_size, enc_seq_len, enc_hid_dim]
        :return: tuple of - (Attentioned hidden state, Alphas)
        """
        # (batch, seq_len)
        attn = self.score(input, context)

        # get source mask according to contex
        context_mask = torch.eq(torch.mean(context, -1), 0)
        attn.masked_fill_(context_mask, float('-inf'))
        attn_dist = self.softmax(attn) # [B, seq_len]
        
        # [B, enc_hid_dim]
        attn_context = torch.bmm(context.permute(0, 2, 1), attn_dist.unsqueeze(2)).squeeze(2)

        return attn_context, attn_dist


class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim, hidden_dim, attn_method, attn_dim, dropout):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn = Attention(attn_method, hidden_dim, hidden_dim, attn_dim)
        self.calc_attentive_x = nn.Linear(hidden_dim + embedding_dim, hidden_dim)
        # To calculate pgen, we use [context_vector, state.c, state.h, x]
        self.dropout = nn.Dropout(p=dropout)
        self.calc_pgen = nn.Linear(4 * hidden_dim, 1)


    def lstm_step(self, x, state, context):
        """
        Recurrence step function

        :param Tensor x: Input at time t, x = [batch_size, embed_dim]
        :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1, (h0, c0) = [batch_size, hid_dim]
        :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
        """

        # Regular LSTM
        h, c = state
        gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
        input, forget, cell, out = gates.chunk(4, 1)

        input = F.sigmoid(input)
        forget = F.sigmoid(forget)
        cell = F.tanh(cell)
        out = F.sigmoid(out)

        c_t = (forget * c) + (input * cell)
        h_t = out * F.tanh(c_t)
        # h_t = [batch_size, hid_dim]

        # Attention section
        attn_ctx, attn_dist = self.attn(h_t, context)
        # attn_context = [batch_size, enc_hid_dim]

        return h_t, c_t, attn_ctx, attn_dist

    def output_step(self, x, h_t, c_t, attn_ctx):
        # refer to https://github.com/abisee/pointer-generator/blob/0cdcaeeaf8f42d4d64ec2ed09eb2f0158cd0db8f/attention_decoder.py#L166
        # https://arxiv.org/pdf/1704.04368.pdf Eqn.(8) 
        out = self.hidden_out(torch.cat((attn_ctx, h_t), 1))
        calc_input = torch.cat([attn_ctx, h_t, c_t, x], dim=1)
        p_gen = self.calc_pgen(calc_input)
        p_gen = F.sigmoid(p_gen)
        return out, p_gen
    

    def forward(self, embedded_inputs, state, context):
        """
        Decoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net, [batch_size, seq_len, embed_dim]
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)
        assert context.size(2) == self.hidden_dim, 'Please make sure the encoder output hidden dim = decoder hidden dim'

        outputs = [] # a list of comb_h_t
        attn_dists = []
        p_gens = []
        
        decoder_inputs = [each.squeeze(1) for each in torch.split(embedded_inputs, 1, dim=1)]
        # Recurrence loop
        for i in range(input_length):
            h_t, c_t, attn_ctx, attn_dist = self.lstm_step(decoder_inputs[i], state, context)
            attn_ctx = self.dropout(attn_ctx)
            x = self.calc_attentive_x(torch.cat([attn_ctx, decoder_inputs[i]], dim=1))
            x = self.dropout(x)
            output, p_gen = self.output_step(x, h_t, c_t, attn_ctx)
            state = (h_t, c_t)
            #comb_h_t = F.tanh(self.hidden_out(torch.cat((attn_ctx, h_t), 1)))
            attn_dists.append(attn_dist)
            outputs.append(output)
            p_gens.append(p_gen)

        return outputs, state, attn_dists, p_gens


class PointerNet(nn.Module):
    """
    Pointer-Net
    """

    def __init__(self, hps, word_emb):
        """
        Initiate Pointer-Net

        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNet, self).__init__()
        self.hps = hps
        self.word_emb_mat = np.stack(word_emb, axis=0)
        self.vocab_size = len(word_emb)
        self.embedding_dim = hps.embedding_dim
        self.bidir = hps.bidir
        self.encoder_hidden_dim = hps.hidden_dim * 2 if self.bidir else hps.hidden_dim
        self.decoder_hidden_dim = self.encoder_hidden_dim
        self.n_layers = 1
        self.attn_type = hps.attn_type

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(self.word_emb_mat))
        self.embedding.weight.requires_grad = False
        self.paragraph_encoder = Encoder(self.embedding_dim,
                                         hps.hidden_dim,
                                         self.n_layers,
                                         hps.dropout,
                                         self.bidir)
        self.answer_encoder = Encoder(self.encoder_hidden_dim+self.embedding_dim, 
                                      hps.hidden_dim,
                                      self.n_layers, 
                                      hps.dropout,
                                      self.bidir)
        self.L = nn.Parameter(torch.FloatTensor(self.encoder_hidden_dim, self.encoder_hidden_dim)) 
        self.W0 = nn.Parameter(torch.FloatTensor(self.encoder_hidden_dim, self.decoder_hidden_dim)) 
        self.b0 = nn.Parameter(torch.FloatTensor(self.decoder_hidden_dim)) 

        nn.init.uniform_(self.L, -0.1, 0.1)
        nn.init.uniform_(self.W0, -0.1, 0.1)
        nn.init.uniform_(self.b0, -0.1, 0.1)

        self.decoder = Decoder(self.embedding_dim,
                               self.decoder_hidden_dim, 
                               attn_method=hps.attn_type,
                               attn_dim=self.decoder_hidden_dim,
                               dropout=hps.dropout)
        
        # output layer
        self.output = nn.Linear(self.decoder_hidden_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def encoder_forward(self, paragraph_inputs, answer_positions):
        para_length, _ = length_and_mask(paragraph_inputs)

        embedded_para_inputs = self.embedding(paragraph_inputs)

        encoder_para_hidden0 = self.paragraph_encoder.initialize_hidden(embedded_para_inputs)

        # embedded_para_inputs & para_length has sorted
        para_encoder_outputs, para_encoder_hidden = self.paragraph_encoder(embedded_para_inputs, para_length, encoder_para_hidden0)
        ht_ans = self.answer_encoding(embedded_para_inputs, para_encoder_outputs, answer_positions) 
        decoder_h0 = self.combine_hidden(ht_ans, para_encoder_outputs)
        decoder_c0 = torch.zeros_like(decoder_h0).cuda()
        init_state = (decoder_h0, decoder_c0)
        return para_encoder_outputs, init_state
        

    def forward(self, paragraph_inputs, question_inputs, answer_positions,
                paragraph_inputs_extend_vocab=None, max_para_oovs=None):
        """
        PointerNet - Forward-pass

        :param Tensor inputs: Input sequence
        :return: Pointers probabilities and indices
        """
        if self.hps.pointer_gen:
            assert (paragraph_inputs_extend_vocab is not None) and (max_para_oovs is not None)
            self.paragraph_inputs_extend_vocab = paragraph_inputs_extend_vocab
            self.max_para_oovs = max_para_oovs

        para_encoder_outputs, init_state = self.encoder_forward(paragraph_inputs, answer_positions)
        embedded_ques_inputs = self.embedding(question_inputs)
        

        outputs, state, attn_dists, self.p_gens = self.decoder(embedded_ques_inputs,
                                                               init_state,
                                                               para_encoder_outputs)
        # compute vocabulary distribution
        vocab_scores = []
        for i, out in enumerate(outputs):
            vocab_scores.append(self.output(out))
        vocab_dists = [self.softmax(s) for s in vocab_scores]
         
        # compute final distribution
        if self.hps.pointer_gen:
            final_dists = self.calc_final_dist(vocab_dists, attn_dists)
        else:
            final_dists = vocab_dists

        return vocab_scores, vocab_dists, attn_dists, final_dists


    def answer_encoding(self, embedded_para_inputs, para_outputs, answer_positions):
        answer_length = answer_positions[:,1] - answer_positions[:,0]
        answer_hidden = torch.zeros_like(para_outputs).cuda()
        embedded_answer = torch.zeros_like(embedded_para_inputs).cuda()
        for i, (start, end) in enumerate(answer_positions):
            start, end = start.item(), end.item()
            answer_hidden[i,0:answer_length[i].item()] = para_outputs[i,start:end]
            embedded_answer[i,0:answer_length[i].item()] = embedded_para_inputs[i,start:end]
        comb_h = torch.cat((answer_hidden, embedded_answer), dim=-1)
        # 根据ans length排序, 排序后才能进行lstm
        sorted_ans_len, sorted_ans_indices = torch.sort(answer_length, descending=True)
        sorted_comb_h = comb_h[sorted_ans_indices]
        _, (sorted_ht_ans, _) = self.answer_encoder(sorted_comb_h, sorted_ans_len, self.answer_encoder.initialize_hidden(sorted_comb_h))
        sorted_ht_ans = self.answer_encoder.fix_hidden(sorted_ht_ans).squeeze(0)
        # ans encoding后必须还原成原来的顺序
        ht_ans = restore_order(sorted_ht_ans, sorted_ans_indices) 
        return ht_ans
    

    def combine_hidden(self, answer_hidden, paragraph_hidden):
        """
        refer to the paper of `` in Sec 3.1, the first function
        @params:
            - encode_answer: ha or ca, each of shape = [batch_size, hid_dim*n_dir]
            - encode_paragraph: [batch_size, seq_len, hid_dim*n_dir]
        """
        r = torch.mm(answer_hidden, self.L) + torch.mean(paragraph_hidden, 1)
        return F.tanh(torch.mm(r, self.W0) + self.b0)
   

    def calc_final_dist(self, vocab_dists, attn_dists):
        vocab_dists = [p_gen * dist for (p_gen,dist) in zip(self.p_gens, vocab_dists)]
        attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(self.p_gens, attn_dists)]
        batch_size = vocab_dists[0].size(0)
        if self.max_para_oovs > 0:
            extra_zeros = torch.zeros((batch_size, self.max_para_oovs)).cuda()
            vocab_dists_extended = [torch.cat([dist, extra_zeros], dim=1) for dist in vocab_dists]
        else:
            vocab_dists_extended = vocab_dists
        extended_vsize = self.vocab_size + self.max_para_oovs
        attn_dists_projected =  [torch.zeros(batch_size, extended_vsize).cuda().scatter_add_(1, self.paragraph_inputs_extend_vocab, copy_dist) for copy_dist in attn_dists]
        
        final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]
        return final_dists


def length_and_mask(input_tensor):
    mask = torch.eq(input_tensor, 0)
    length = input_tensor.size(1) - torch.sum(mask, dim=1)
    return length, mask


def restore_order(sorted_data, sorted_indices):
    _, raw_indices = torch.sort(sorted_indices, dim=0)
    return sorted_data[raw_indices]


