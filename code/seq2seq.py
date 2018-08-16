import data
import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from utils import length_and_mask


class Seq2Seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio, volatile
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim, enc_n_layers, 
                 dec_n_layers, dec_max_length,
                 dropout_ratio,
                 use_attention=False,
                 use_copy=False,
                 pretrain_w2v=None, update_embedding=False):
        super(Seq2Seq, self).__init__()
        self.use_attention = use_attention
        self.use_copy = use_copy
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrain_w2v is not None:
            self.embedding.weight = nn.Parameter(torch.from_numpy(pretrain_w2v).float())
        self.embedding.weight.requires_grad = update_embedding
        self.encoder = Encoder(self.embedding,
                               hidden_dim,
                               enc_n_layers,
                               bidir=True,
                               dropout_p=dropout_ratio,
                               input_dropout_p=dropout_ratio)

        self.decoder = Decoder(self.embedding,
                               hidden_dim*2,
                               dec_n_layers,
                               dec_max_length,
                               data.GO_ID, data.EOS_ID,
                               dropout_p=dropout_ratio,
                               input_dropout_p=dropout_ratio,
                               use_attention=use_attention,
                               use_copy=use_copy)

    def forward(self, batch):
        encoder_input_variable = torch.tensor(batch.enc_batch, dtype=torch.long, requires_grad=False, device=torch.device('cuda'))
        dec_batch = batch.dec_batch
        decoder_input_variable = torch.tensor(dec_batch, dtype=torch.long, requires_grad=False, device=torch.device('cuda'))
        encoder_input_lengths, encoder_input_mask = length_and_mask(encoder_input_variable)

        encoder_outputs, encoder_hidden = self.encoder(encoder_input_variable, encoder_input_lengths)
        
        enc_inputs_extend_vocab_variable = None
        max_enc_oov = None
        if self.use_copy:
            enc_inputs_extend_vocab_variable = torch.tensor(batch.enc_batch_extend_vocab, dtype=torch.long, 
                    requires_grad=False, device=torch.device('cuda'))
            max_enc_oov = batch.max_enc_oovs

        result = self.decoder(inputs=decoder_input_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              encoder_mask=encoder_input_mask,
                              enc_inputs_extend_vocab=enc_inputs_extend_vocab_variable,
                              max_enc_oov=max_enc_oov)
        return result
