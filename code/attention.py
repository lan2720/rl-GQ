import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        # concat type
        self.dim = dim
        self.linear_context = nn.Linear(dim, dim, bias=False)
        self.linear_query = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim*2, dim, bias=True)

    def forward(self, output, context, mask):
        # refer to OpenNMT-py
        # https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/global_attention.py#L121
        dim = self.dim
        src_batch, src_len, src_dim = context.size()
        tgt_batch, tgt_len, tgt_dim = output.size()
        assert src_batch == tgt_batch
        assert src_dim == tgt_dim
        assert src_dim == self.dim

        wq = self.linear_query(output.contiguous().view(-1, dim))
        wq = wq.view(tgt_batch, tgt_len, 1, dim)
        wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

        uh = self.linear_context(context.contiguous().view(-1, dim))
        uh = uh.view(src_batch, 1, src_len, dim)
        uh = uh.expand(src_batch, tgt_len, src_len, dim)

        wquh = F.tanh(wq + uh)
        
        score = self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)
        
        # mask = [batch_size, tgt_len, src_len]
        mask = mask.unsqueeze(1).expand_as(score).contiguous().view(src_batch*tgt_len, src_len)
        #score.data.masked_fill_(mask, -float('inf'))
        score = score.view(tgt_batch*tgt_len, src_len)
        max_by_row = torch.max(score, dim=1, keepdim=True)[0]
        #attn = F.softmax(score-max_by_row, dim=1).view(tgt_batch, tgt_len, src_len)
        attn = torch.exp(score-max_by_row) * (1.0 - mask.float())
        sum_attn = torch.sum(attn, dim=1, keepdim=True)
        #zero_mask = torch.eq(attn, 0.)
        attn = attn/sum_attn
        #attn.masked_fill_(zero_mask, 0.)
        attn = attn.view(tgt_batch, tgt_len, src_len)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = self.linear_out(combined.view(-1, 2 * dim)).view(tgt_batch, -1, dim)
        # output ~ [ht, attn_ctx]

        return output, attn
