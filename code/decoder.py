import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention


class Decoder(nn.Module):

    KEY_ATTN_SCORE = 'attention_score'
    KEY_COPY_PROB = 'copy_prob'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, embedding, 
                 hidden_dim, 
                 n_layers,
                 max_len, sos_id, eos_id,
                 input_dropout_p=0, dropout_p=0, 
                 use_attention=False, use_copy=False):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.hidden_dim = hidden_dim
        self.embed_dim = embedding.embedding_dim
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.lstm = nn.LSTM(self.embed_dim,
                            hidden_dim,
                            n_layers,
                            batch_first=True,
                            dropout=dropout_p,
                            bidirectional=False)

        self.output_size = self.embedding.num_embeddings
        self.max_length = max_len # for when not teacher forcing
        self.use_attention = use_attention
        self.use_copy = use_copy
        self.sos_id = sos_id
        self.eos_id = eos_id

        if use_attention:
            self.attention = Attention(self.hidden_dim)
        if use_copy:
            self.copy = nn.Linear(self.hidden_dim+self.embed_dim, 1)

        self.out = nn.Linear(self.hidden_dim, self.output_size)

    def forward_step(self, input_var, hidden, encoder_outputs, encoder_mask):
        batch_size = input_var.size(0)
        dec_len = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        
        output, hidden = self.lstm(embedded, hidden)

        attn = None
        p_copy = None
        if self.use_attention:
            # output ~ [ht, attn_ctx]
            output, attn = self.attention(output, encoder_outputs, encoder_mask)
        if self.use_copy:
            p_copy = self.copy(torch.cat((output, embedded), dim=2).view(batch_size*dec_len, -1)).squeeze(1).view(batch_size, dec_len)

        predicted_softmax = F.softmax(self.out(output.contiguous().view(-1, self.hidden_dim)), dim=1).view(batch_size, dec_len, -1)
        return predicted_softmax, hidden, attn, p_copy

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, encoder_mask=None,
                enc_inputs_extend_vocab=None, max_enc_oov=None):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[Decoder.KEY_ATTN_SCORE] = list()
        if self.use_copy:
            assert max_enc_oov is not None, "Please provide max_enc_oovs when use_copy=True"
            ret_dict[Decoder.KEY_COPY_PROB] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs, encoder_mask)
        decoder_hidden = self._init_state(encoder_hidden)

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def calc_final_dist(step_vocab, step_attn, step_copy,
                            enc_inputs_extend_vocab, max_enc_oov):
            """

            :param self:
            :param step_vocab: [batch_size, vocab_size]
            :param step_attn: [batch_size, max_enc_len]
            :param step_copy: [batch_size, 1]
            :param enc_inputs_extend_vocab: [batch_size, max_enc_len]
            :param max_enc_oov: scalar
            :return:
            """
            batch_size, vocab_size = step_vocab.size()
            vocab_dist = (1.0 - step_copy) * step_vocab
            attn_dist = step_copy * step_attn

            if max_enc_oov > 0:
                extra_zeros = torch.zeros((batch_size, max_enc_oov)) #, device=torch.device('cuda'))
                vocab_dist_extended = torch.cat([vocab_dist, extra_zeros], dim=1)
            else:
                vocab_dist_extended = vocab_dist
            extended_vsize = vocab_size + max_enc_oov
            attn_dist_projected = torch.zeros(batch_size, extended_vsize#,
                                              #device=torch.device('cuda')
                                              ).scatter_add_(1, enc_inputs_extend_vocab, attn_dist)

            final_dist = vocab_dist_extended + attn_dist_projected
            return final_dist

        def greedy(step, step_output, step_attn, step_copy,
                   enc_inputs_extend_vocab, max_enc_oov):
            if self.use_copy:
                assert step_copy is not None, "Please provide step_copy when use_copy=True"
                assert enc_inputs_extend_vocab is not None, "Please provide enc_inputs_extend_vocab when use_copy=True"
                assert max_enc_oov is not None, "Please provide max_enc_oov when use_copy=True"
                step_final = calc_final_dist(step_output, step_attn, step_copy, enc_inputs_extend_vocab, max_enc_oov)
            else:
                step_final = step_output
            decoder_outputs.append(step_final)
            if self.use_attention:
                assert step_attn is not None, "Please provide step_attn when use_attn=True"
                ret_dict[Decoder.KEY_ATTN_SCORE].append(step_attn)
            if self.use_copy:
                assert step_copy is not None, "Please provide step_copy when use_copy=True"
                ret_dict[Decoder.KEY_COPY_PROB].append(step_copy)

            symbols = decoder_outputs[-1].topk(k=1, dim=1)[1] # [batch_size, k]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        decoder_output, decoder_hidden, attn, p_copy = self.forward_step(inputs, decoder_hidden,
                                                                         encoder_outputs, encoder_mask)

        for di in range(decoder_output.size(1)):
            step_output = decoder_output[:, di, :]
            if attn is not None:
                step_attn = attn[:, di, :]
            else:
                step_attn = None
            if p_copy is not None:
                step_copy = p_copy[:, di].unsqueeze(1)
            else:
                step_copy = None
            greedy(di, step_output, step_attn, step_copy, enc_inputs_extend_vocab, max_enc_oov)

        ret_dict[Decoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[Decoder.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        if h.size(0) > 1:
            h = h[-1,:,:].unsqueeze(0)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, encoder_mask):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
            if encoder_mask is None:
                raise ValueError("Argument encoder_mask cannot be None when attention is used.")

        # inference batch size
        if inputs is None:
            if encoder_hidden is None:
                batch_size = 1
            else:
                batch_size = encoder_hidden[0].size(1)
        else:
            batch_size = inputs.size(0)

        # set default input and max decoding length
        if inputs is None:
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1)

        return inputs, batch_size, max_length

def main():
    vocab_size = 10
    embed_dim = 3
    hidden_dim = 4
    n_layers = 1
    max_dec_len = 8
    max_enc_len = 7
    sos_id = 2
    eos_id = 3
    dropout_p = 0.0
    batch_size = 2
    use_attention = True
    use_copy = True
    embedding = nn.Embedding(vocab_size, embed_dim)
    decoder = Decoder(embedding, hidden_dim, n_layers, max_dec_len, sos_id, eos_id, dropout_p, dropout_p,
                      use_attention=use_attention, use_copy=use_copy)
    inputs = torch.tensor([[sos_id, 4,2,5,6], [sos_id, 2,7,5,3]])
    encoder_outputs = torch.rand(2, max_enc_len, hidden_dim)
    encoder_mask = torch.tensor([[0,0,0,0,1,1,1], [0,0,0,0,0,1,1]], dtype=torch.uint8)
    encoder_inputs_extend_vocab = torch.tensor([[3,4,2,13,5,7,1], [14,5,3,6,2,7,9]])
    max_enc_oov = 5
    decoder_outputs, decoder_hidden, ret_dict = decoder(inputs, encoder_hidden=None, encoder_outputs=encoder_outputs,
                                                 encoder_mask=encoder_mask,
                                                 enc_inputs_extend_vocab=encoder_inputs_extend_vocab, max_enc_oov=max_enc_oov)
    print('dec out: softmax', decoder_outputs)
    # print('dec hid: final', decoder_hidden)
    print('dict:', ret_dict)

if __name__ == '__main__':
    main()
