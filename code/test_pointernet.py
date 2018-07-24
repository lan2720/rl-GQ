from PointerNet import PointerNet
from data import Vocab
from batcher import batcher#NewBatcher
import torch
import argparse
import numpy as np
from torch import optim
import time
import pickle
# python test_pointernet.py --data_path=../data/squad-v1/dev_raw.json --word_count_path=../data/squad-v1/word_counter.json --glove_path=/home/jiananwang/data/glove/glove.840B.300d.txt --pointer_gen --bidir

parser = argparse.ArgumentParser(description='arguments.')
parser.add_argument('--mode', type=str, default='train', choices=['eval', 'decode', 'train'],
                    help='the data path to load raw data')
parser.add_argument('--data_path', type=str, default='../data/squad-v1/dev_raw.json',
                    help='the data path to load raw data')
parser.add_argument('--word_count_path', type=str, default='../data/squad-v1/dev_raw.json',
                    help='the data path to load raw data')
parser.add_argument('--glove_path', type=str, default='../data/squad-v1/dev_raw.json',
                    help='the data path to load raw data')
parser.add_argument('--batch_size', type=int, default=100,
                    help='the num of examples in one batch')
parser.add_argument('--max_enc_steps', type=int, default=65,
                    help='the num of examples in one batch')
parser.add_argument('--max_dec_steps', type=int, default=65,
                    help='the num of examples in one batch')
parser.add_argument('--single_pass', default=False, action='store_true',
                    help='whether pass the only one example each time') # only True when decoding
parser.add_argument('--pointer_gen', default=False, action='store_true',
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--hidden_dim', type=int, default=100,
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--embedding_dim', type=int, default=300,
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--attn_type', type=str, default='concat', choices=['dot', 'general', 'concat'],
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--bidir', default=False, action='store_true',
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--print_every', type=int, default=1,
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--dropout', type=float, default=0.,
                    help='whether to use pointer mechanism') # only True when decoding
hps = parser.parse_args()


def mask_and_avg(values, padding_mask):
    """
    values: a list of (batch_size), list length = max_dec_steps
    padding_mask: (batch_size, max_dec_steps)
    
    Return:
        a scalar
    """
    lens = torch.sum(padding_mask, 1)
    values_per_step = [v * padding_mask[:,step] for step, v in enumerate(values)] # a list of (batch_size,)
    values_per_ex = sum(values_per_step)/lens # (batch_size,)/(batch_size,)-> (batch_size,)
    return torch.mean(values_per_ex)


def main():
    vocab = Vocab(hps.word_count_path, hps.glove_path, hps.embedding_dim)

    net = PointerNet(hps, np.stack(vocab.emb_mat, axis=0))
    net = net.cuda()

    data_batcher = batcher(hps.data_path, vocab, hps, hps.single_pass)
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(model_parameters)

    loss_track = []
    global_step = 0
    while True:
        start = time.time()
        batch = next(data_batcher)
        #batch = pickle.load(open('one_batch.pkl', 'rb'))
        paragraph_tensor = torch.tensor(batch.enc_batch, dtype=torch.int64, requires_grad=False).cuda()
        question_tensor = torch.tensor(batch.dec_batch, dtype=torch.int64, requires_grad=False).cuda()
        answer_position_tensor = torch.tensor(batch.ans_indices, dtype=torch.int64, requires_grad=False).cuda()
        target_tensor = torch.tensor(batch.target_batch, dtype=torch.int64, requires_grad=False).cuda()
        
        paragraph_batch_extend_vocab = None
        max_para_oovs = None
        if hps.pointer_gen:
            paragraph_batch_extend_vocab = torch.tensor(batch.enc_batch_extend_vocab, dtype=torch.int64, requires_grad=False).cuda()
            max_para_oovs = batch.max_para_oovs
        
        vocab_scores, vocab_dists, attn_dists, final_dists = net(paragraph_tensor, question_tensor, answer_position_tensor,
                                                                 paragraph_batch_extend_vocab, max_para_oovs)
        
        optimizer.zero_grad()
        dec_padding_mask = torch.ne(target_tensor, 0).float().cuda()
        if hps.pointer_gen:
            loss_per_step = []
            for dec_step, dist in enumerate(final_dists):
                # dist = [batch_size, extended_vsize]
                targets = target_tensor[:,dec_step]
                gold_probs = torch.gather(dist, 1, targets.unsqueeze(1)).squeeze()
                losses = -torch.log(gold_probs)
                loss_per_step.append(losses) # a list of [batch_size,]
            loss = mask_and_avg(loss_per_step, dec_padding_mask)
        else:
            # a list of dec_max_len (vocab_scores)
            loss_batch_by_step = F.cross_entropy(torch.cat(vocab_scores, dim=1).reshape(-1, vocab.size()), target_tensor.reshape(-1), size_average=False, reduce=False)
            # loss [batch_size*dec_max_len,]
            loss = torch.sum(loss_batch_by_step * dec_padding_mask.reshape(-1))/torch.sum(dec_padding_mask)
        loss_track.append(loss.item())
        global_step += 1

        loss.backward()
        optimizer.step()
        if global_step % hps.print_every == 0:
            print('Step {:>10}: ave loss: {:>10.4f}, speed: {:.4f}/case'.format(global_step, sum(loss_track)/len(loss_track), (time.time()-start)/hps.batch_size))
            loss_track = []
        #break

if __name__ == '__main__':
    main()
#print('attn dist:') 
#print(attn_dists[0].data.numpy().tolist()) #[B, seq_len]
#print('enc extend vocab:')
#print(batch.enc_batch_extend_vocab.tolist())
#print('enc len:')
#print(batch.enc_lens)
#print('final dist:')
#print(final_dists)

