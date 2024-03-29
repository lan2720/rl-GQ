from seq2seq import Seq2Seq
import os
import time
import math
import data
import torch
import torch.optim as optim
import torch.nn as nn
from batcher import Batcher, decoding
from decoder import Decoder
from loss import loss

def main():
    data_path = '/home/jiananwang/rl-QG/data/squad-v1'
    word_count_path = os.path.join(data_path, 'word_counter.json')
    glove_path = '/home/jiananwang/data/glove/glove.840B.300d.txt'
    embedding_dim = 300
    max_vocab_size = 50000
    embedding_dict_file = os.path.join(data_path, 'emb_dict_%d.pkl' % max_vocab_size)

    vocab = data.Vocab(word_count_path, glove_path, embedding_dim, max_vocab_size, embedding_dict_file)
    train_file = os.path.join(data_path, 'train_raw_100.json')
    #dev_file = os.path.join(data_path, 'dev_raw.json')#'dev_raw.json')

    hidden_dim = 100
    p = 0.


    max_enc_steps = 65
    max_dec_steps = 65
    use_attention = False #True
    use_copy = False #True
    dynamic_vocab = True if use_copy else False
    update_embedding = False#True
    batch_size = 10

    model = Seq2Seq(vocab.size(),
                    embedding_dim,
                    hidden_dim,
                    enc_n_layers=2,
                    dec_n_layers=1,
                    dec_max_length=max_dec_steps,
                    dropout_ratio=p,
                    use_attention=use_attention,
                    use_copy=use_copy,
                    pretrain_w2v=vocab.emb_mat,
                    update_embedding=update_embedding)

    if torch.cuda.is_available():
        model = model.cuda()

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    print('the number of parameters in model:', sum(p.numel() for p in model_parameters))
    optimizer = optim.Adagrad(model_parameters)#, lr=1.)

    train_data_batcher = Batcher(train_file, vocab, batch_size,
                                 max_enc_steps, max_dec_steps,
                                 mode='train', dynamic_vocab=dynamic_vocab)
    #dev_data_batcher = Batcher(dev_file, vocab, hps, hps.single_pass)

    num_epoch = 50
    for i in range(num_epoch):
        train_data_batcher.setup()
        while True:
            try:
                start = time.time()
                batch = train_data_batcher.next_batch()
            except StopIteration:
                print('epoch %d finish' % i)
                break
            
            decoder_outputs, _, ret_dict = model(batch)
            # list -> [batch_size, dec_len, vocab_size]
            vocab_dist = torch.stack(decoder_outputs, dim=1)

            targets = torch.tensor(batch.target_batch, dtype=torch.int64, requires_grad=False, device=torch.device('cuda'))
            targets_mask = torch.eq(targets, data.PAD_ID)
            
            attn_dist = None
            p_copy = None
            if use_attention:
                attn_dist = torch.stack(ret_dict[Decoder.KEY_ATTN_SCORE], dim=1)
            if use_copy:
                p_copy = torch.cat(ret_dict[Decoder.KEY_COPY_PROB], dim=1)
            
            if use_copy:
                enc_batch_extend_vocab_variable = torch.tensor(batch.enc_batch_extend_vocab, dtype=torch.long, device=torch.device('cuda'))
                raw_loss = loss(targets, vocab_dist, targets_mask, use_copy=use_copy,
                                attn_dist=attn_dist, p_copy=p_copy, enc_batch_extend_vocab=enc_batch_extend_vocab_variable, max_enc_oovs=batch.max_enc_oovs)
            else:
                raw_loss = loss(targets, vocab_dist, targets_mask, use_copy=use_copy,
                                attn_dist=attn_dist, p_copy=None, enc_batch_extend_vocab=None, max_enc_oovs=None)

            # loss by case = [batch_size, ]
            loss_by_case = torch.sum(raw_loss, dim=1)
            ave_loss_by_case = torch.mean(loss_by_case)
            ave_loss_by_char = torch.sum(raw_loss)/(targets_mask.size(0)*targets_mask.size(1) - torch.sum(targets_mask)).float()
            char_ppl = math.exp(ave_loss_by_char.item())
            print('char ppl: %.4f, ave loss by case: %.4f' % (char_ppl, ave_loss_by_case.item()))
            ave_loss_by_case.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            seqs = torch.cat(ret_dict[Decoder.KEY_SEQUENCE], dim=1)
            enc_oovs_batch = None
            if use_copy:
                for s, enc_oovs in zip(seqs, batch.enc_oovs_batch):
                    words = decoding(s.data.tolist(), vocab, enc_oovs)
                    print(' '.join(words))
            else:
                for s in seqs:
                    words = decoding(s.data.tolist(), vocab)
                    print(' '.join(words))


if __name__ == '__main__':
    main()
