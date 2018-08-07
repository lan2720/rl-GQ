from data import Vocab
from batcher import Batcher, decoding
import torch
import torch.nn as nn
import argparse
import numpy as np
from torch import optim
import time
import os
import sys
import pickle
import math
import copy
from nltk.translate.bleu_score import corpus_bleu
from multi_bleu import bleu
import utils
import torch.nn.functional as F
from seq2seq import Seq2Seq
from decoder import Decoder
import data
# python run_question_generator.py --data_path=../data/squad-v1/ --word_count_path=../data/squad-v1/word_counter.json --glove_path=/home/jiananwang/data/glove/glove.840B.300d.txt --pointer_gen --bidir
# python run_question_generator.py --data_path=../data/squad-v1/ --word_count_path=../data/squad-v1/word_counter.json --glove_path=/home/jiananwang/data/glove/glove.840B.300d.txt --pointer_gen --bidir --self_critic
# python run_question_generator.py --exp_dir=../exp/pretrain --data_path=../data/squad-v1/ --word_count_path=../data/squad-v1/word_counter.json --glove_path=/home/jiananwang/data/glove/glove.840B.300d.txt --maxium_likelihood --pointer_gen --bidir --save


parser = argparse.ArgumentParser(description='arguments.')
parser.add_argument('--mode', type=str, default='train', choices=['eval', 'decode', 'train'],
                    help='the data path to load raw data')
parser.add_argument('--data_path', type=str, default='../data/squad-v1/',
                    help='the data path to load raw data')
parser.add_argument('--word_count_path', type=str, default='../data/squad-v1/dev_raw.json',
                    help='the data path to load raw data')
parser.add_argument('--glove_path', type=str, default='../data/squad-v1/dev_raw.json',
                    help='the data path to load raw data')
parser.add_argument('--batch_size', type=int, default=72,
                    help='the num of examples in one batch')
parser.add_argument('--num_epoch', type=int, default=50,
                    help='the num of epoch to train')
parser.add_argument('--max_enc_steps', type=int, default=65,
                    help='the num of examples in one batch')
parser.add_argument('--max_dec_steps', type=int, default=65,
                    help='the num of examples in one batch')
parser.add_argument('--max_vocab_size', type=int, default=50000,
                    help='the num of examples in one batch')
parser.add_argument('--single_pass', default=False, action='store_true',
                    help='whether pass the only one example each time') # only True when decoding
parser.add_argument('--pointer_gen', default=False, action='store_true',
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--self_critic', default=False, action='store_true',
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--maxium_likelihood', default=False, action='store_true',
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--hidden_dim', type=int, default=100,
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--embedding_dim', type=int, default=300,
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--update_embedding', default=False, action='store_true',
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--use_attention', default=False, action='store_true',
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--attn_type', type=str, default='concat', choices=['dot', 'general', 'concat'],
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--dynamic_vocab', default=False, action='store_true',
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--reward_metric', type=str, default='bleu', choices=['bleu'],
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--bidir', default=False, action='store_true',
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--print_every', type=int, default=100,
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--dropout', type=float, default=0.,
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.,
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--norm_limit', type=float, default=2.,
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--early_stopping_from', type=int, default=9,
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--save', default=False, action='store_true',
                    help='whether to use pointer mechanism') # only True when decoding
parser.add_argument('--exp_dir', type=str,
                    help='whether to use pointer mechanism') # only True when decoding
hps = parser.parse_args()



def mask_and_avg(values, padding_mask, batch_average=True, step_average=True):
    """
    values: a list of (batch_size), list length = max_dec_steps
    padding_mask: (batch_size, max_dec_steps)
    
    Return:
        a scalar
    """
    lens = torch.sum(padding_mask, 1)
    values_per_step = [v * padding_mask[:,step] for step, v in enumerate(values)] # a list of (batch_size,)
    values_per_ex = sum(values_per_step)
    if step_average:
        values_per_ex = values_per_ex/lens # (batch_size,)/(batch_size,)-> (batch_size,)
    if batch_average:
        return torch.mean(values_per_ex)
    else:
        return values_per_ex


def run_eval(batcher, model):
    model.eval()
    loss_track = []
    while True:
        try:
            batch = batcher.next_batch()
        except StopIteration:
            break
        paragraph_tensor = torch.tensor(batch.enc_batch, dtype=torch.int64, requires_grad=False).cuda()
        question_tensor = torch.tensor(batch.dec_batch, dtype=torch.int64, requires_grad=False).cuda()
        answer_position_tensor = torch.tensor(batch.ans_indices, dtype=torch.int64, requires_grad=False).cuda()
        target_tensor = torch.tensor(batch.target_batch, dtype=torch.int64, requires_grad=False).cuda()
        
        paragraph_batch_extend_vocab = None
        max_para_oovs = None
        if hps.pointer_gen:
            paragraph_batch_extend_vocab = torch.tensor(batch.enc_batch_extend_vocab, dtype=torch.int64, requires_grad=False).cuda()
            max_para_oovs = batch.max_para_oovs

        vocab_scores, vocab_dists, attn_dists, final_dists = model(paragraph_tensor, question_tensor, answer_position_tensor,
                                                                 paragraph_batch_extend_vocab, max_para_oovs)
        
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
            loss_batch_by_step = F.cross_entropy(torch.stack(vocab_scores, dim=1).reshape(-1, vocab_scores[0].size(1)), target_tensor.reshape(-1), size_average=False, reduce=False)
            # loss [batch_size*dec_max_len,]
            loss = torch.sum(loss_batch_by_step * dec_padding_mask.reshape(-1))/torch.sum(dec_padding_mask)
        loss_track.append(loss.item())

    if len(loss_track) > 0:
        return sum(loss_track)/len(loss_track)
    else:
        raise Exception()


def get_batch_bleu(refs, hyps):
    """
    @params:
        - refs: tensor, [batch_size, max_seq_len], 
        - hyps: tensor, [batch_size, max_seq_len]
    @return:
        a scalar 
    """
    refs = refs.unsqueeze(1)
    if refs.is_cuda:
        refs = refs.cpu()
        hyps = hyps.cpu()
    
    #refs = refs.data.tolist()
    #hyps = hyps.data.tolist()
    
    results = []
    refs_batch = torch.split(refs, 1, dim=0)
    hyps_batch = torch.split(hyps, 1, dim=0)
    for ref, hyp in zip(refs_batch, hyps_batch):
        tmp = bleu(hyp, ref, 2)
        results.append(tmp[0])
    return torch.tensor(results).cuda()


def main():
    embedding_dict_file = os.path.join(os.path.dirname(hps.word_count_path), 'emb_dict_50000.pkl')
    vocab = Vocab(hps.word_count_path, hps.glove_path, hps.embedding_dim, hps.max_vocab_size, embedding_dict_file)
    train_file = os.path.join(hps.data_path, 'train_raw.json')
    dev_file = os.path.join(hps.data_path, 'dev_raw.json')#'dev_raw.json')
    
    if (not os.path.exists(train_file)) \
    or (not os.path.exists(dev_file)):
        raise Exception('train and dev data not exist in data_path, please check')

    if hps.save and not hps.exp_dir:
        raise Exception('please specify exp_dir when you want to save experiment info')
    
    print(vars(hps))
    if hps.save:
        utils.save_hps(hps.exp_dir, hps)
    
    net = Seq2Seq(vocab.size(), hps.embedding_dim, hps.hidden_dim,
                  enc_n_layers=1, dec_n_layers=1, dec_max_length=hps.max_dec_steps,
                  dropout_ratio=hps.dropout, 
                  use_attention=hps.use_attention,
                  embedding=vocab.emb_mat, update_embedding=hps.update_embedding)
    net = net.cuda()

    model_parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
    print('the number of parameters in model:', sum(p.numel() for p in model_parameters))
    optimizer = optim.Adam(model_parameters)
    
    train_data_batcher = Batcher(train_file, vocab, hps.batch_size, 
                                 hps.max_enc_steps, hps.max_dec_steps,
                                 mode=hps.mode, dynamic_vocab=hps.dynamic_vocab)
    #dev_data_batcher = Batcher(dev_file, vocab, hps, hps.single_pass)
    loss = nn.NLLLoss(size_average=False, reduce=False)
    
    global_step = 0
    dev_loss_track = []
    min_dev_loss = math.inf
    net.train()
    for i in range(hps.num_epoch):
        epoch_loss_track = []
        train_data_batcher.setup()
        while True:
            start = time.time()
            try:
                batch = train_data_batcher.next_batch()
                #print('get next batch time:', time.time()-start)
            except StopIteration:
               # do evaluation here, if necessary, to save best model
                #dev_data_batcher.setup()
                #dev_loss = run_eval(dev_data_batcher, net)
                #print("epoch {}: avg train loss: {:>10.4f}, dev_loss: {:>10.4f}".format(i+1, sum(epoch_loss_track)/len(epoch_loss_track), dev_loss))
                #dev_loss_track.append(dev_loss)

                #if i > hps.early_stopping_from:
                #    last5devloss = dev_loss_track[i] + dev_loss_track[i-1] + dev_loss_track[i-2] + dev_loss_track[i-3] + dev_loss_track[i-4]
                #    last10devloss = dev_loss_track[i-5] + dev_loss_track[i-6] + dev_loss_track[i-7] + dev_loss_track[i-8] + dev_loss_track[i-9]
                #    if hps.early_stopping_from and last5devloss >= last10devloss:
                #        print("early stopping by dev_loss!")
                #        sys.exit()
                #
                #if dev_loss < min_dev_loss:
                #    min_dev_loss = dev_loss
                #    if hps.save:
                #        utils.save_model(hps.exp_dir, net, min_dev_loss)
                break


            optimizer.zero_grad()

            decoder_outputs, _, ret_dict = net(batch, hps.teacher_forcing_ratio)
            
            decoder_outputs = torch.stack(decoder_outputs, dim=1)

            targets = torch.tensor(batch.target_batch, dtype=torch.int64, requires_grad=False, device=torch.device('cuda'))
            flatten_targets = targets.view(-1)
            targets_mask = torch.eq(flatten_targets, data.PAD_ID)
            
            raw_loss = loss(decoder_outputs.view(-1, net.decoder.output_size), targets.view(-1))
            raw_loss.masked_fill_(targets_mask, 0.)
            # loss by case = [batch_size, ]
            loss_by_case = torch.sum(raw_loss.view(targets.size()), dim=1)
            ave_loss_by_case = torch.mean(loss_by_case)
            ave_loss_by_char = torch.sum(raw_loss)/(targets_mask.size(0) - torch.sum(targets_mask)).float()
            char_ppl = math.exp(ave_loss_by_char.item())
            
            epoch_loss_track.append(ave_loss_by_char.item()) # a list of [batch_size,]
            
            global_step += 1

            ave_loss_by_char.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=hps.norm_limit)
            optimizer.step()
            #print('time one step:', time.time()-start)
            if (global_step == 1) or (global_step % hps.print_every == 0):
                print('Step {:>5}: ave loss: {:>10.4f}, speed: {:.1f} case/s'.format(global_step, sum(epoch_loss_track)/len(epoch_loss_track), hps.batch_size/(time.time()-start)))
                seqs = torch.cat(ret_dict[Decoder.KEY_SEQUENCE], dim=1)
                for s in seqs[:5]:
                    words = decoding(s.data.tolist(), vocab)
                    print(' '.join(words))
            

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

