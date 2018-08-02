# coding:utf-8
"""
USEAGE:
    CUDA_VISIBLE_DEVICES=5 python inference.py --test_query_file=dataset/weibo/stc_weibo_valid_post --load_path=exp/weibo/pretrain/2018-02-01-11-40-16/  --load_prefix=best
"""
import os
import pickle
import argparse
import logging
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Vocab
import data

from batcher import Batcher

def predict(test_file, vocab, model, hps, output_file=None):
    # data batcher
    hps.batch_size = 1
    test_data_batcher = Batcher(test_file, vocab, hps, hps.single_pass) 
    test_data_batcher.setup()
   
    if output_file:
        fo = open(output_file, 'wb')

    while True:
        try:
            batch = test_data_batcher.next_batch()
        except StopIteration:
            print('---------------------finish-------------------------')
            break
        
        paragraph_tensor = torch.tensor(batch.enc_batch, dtype=torch.int64, requires_grad=False).cuda()
        answer_position_tensor = torch.tensor(batch.ans_indices, dtype=torch.int64, requires_grad=False).cuda()

        encoder_outputs, dec_init_state = model.encoder_forward(paragraph_tensor, answer_position_tensor)

        #hyps, _ = beam_search(model, encoder_outputs, dec_init_state, hps, beam=20, penalty=2.0, nbest=20)
        results = []
        result = greedy(model, encoder_outputs, dec_init_state, hps)
        hyps = [result]
        
        for h in hyps:
            #print('hyp:', h)
            results.append([vocab.id2word(i) for i in h])

        print('*******************************************************')
        print("paragraph:", batch.original_paragraphs[0])
        print("answer:", batch.original_answers[0])
        print("target question:", batch.original_questions[0])
        print("generated questions:\n" + '\n'.join([' '.join(r) for r in results]))
        print('')

    if output_file:
        fo.close()


def greedy(model, context, dec_init_state, hps):
    inp = torch.tensor([data.GO_ID], dtype=torch.int64, requires_grad=False).cuda()
    state = dec_init_state
    result = []
    for i in range(hps.max_dec_steps):
        embedded_inp = model.embedding(inp)
        ds_h, ds_c, attn_ctx, _ = model.decoder.lstm_step(embedded_inp, state, context)
        state = (ds_h, ds_c)
        x = model.decoder.calc_attentive_x(torch.cat([attn_ctx, embedded_inp], dim=1))
        out, _ = model.decoder.output_step(x, ds_h, ds_c, attn_ctx)
        logp = F.log_softmax(out, dim=0)
        #print('logp:', logp.size())
        inp = torch.argmax(logp, dim=1, keepdim=False)
        #print('inp:',inp)
        result.append(inp[0].item())
    return result


def beam_search(model, context, dec_init_state, hps, beam=5, penalty=1.0, nbest=1):
    """
    the code is referred to: 
    https://github.com/dialogtekgeek/DSTC6-End-to-End-Conversation-Modeling/blob/master/ChatbotBaseline/tools/seq2seq_model.py
    """
    go_i = torch.tensor([data.GO_ID], dtype=torch.int64, requires_grad=False).cuda()
    eos_i = torch.tensor([data.EOS_ID], dtype=torch.int64, requires_grad=False).cuda()
    embedded_go = model.embedding(go_i)
    embedded_eos = model.embedding(eos_i)
    #eos_i = Variable(torch.LongTensor([[utils.EOS_ID]]), requires_grad=False)

    ds_h, ds_c, attn_ctx, _ = model.decoder.lstm_step(embedded_go, dec_init_state, context)
    ds = [embedded_go, ds_h, ds_c, attn_ctx]
    hyplist = [([], 0., ds)]
    best_state = None
    comp_hyplist = []
    for l in range(hps.max_dec_steps):
        new_hyplist = []
        argmin = 0
        for seq, lp, st in hyplist:
            # state is [x, ht, ct, attn_ctx]
            embedded_x, ht, ct, attn_ctx = st
            x = model.decoder.calc_attentive_x(torch.cat([attn_ctx, embedded_x], dim=1))
            out, _ = model.decoder.output_step(x, ht, ct, attn_ctx)
            out = out.squeeze()
            #[vocab_size,]
            logp = F.log_softmax(out, dim=0)
            lp_vec = logp.cpu().data.numpy() + lp
            if l > 0:
                new_lp = lp_vec[data.EOS_ID] + penalty*(len(seq)+1)
                new_ht, new_ct, new_attn_ctx, _ = model.decoder.lstm_step(embedded_eos, st[1:3], context)
                #x = model.decoder.calc_attentive_x(torch.cat([new_attn_ctx, embedded_eos], dim=1))
                new_st = [embedded_eos, new_ht, new_ct, new_attn_ctx]
                comp_hyplist.append((seq, new_lp))
                if best_state is None or best_state[0] < new_lp:
                    best_state = (new_lp, new_st)
            
            for o in np.argsort(lp_vec)[::-1]:
                if o == data.UNK_ID or o == data.EOS_ID:
                    continue
                new_lp = lp_vec[o]
                o_var = torch.tensor([o], dtype=torch.int64, requires_grad=False).cuda()
                embedded_o = model.embedding(o_var)
                if len(new_hyplist) == beam:
                    if new_hyplist[argmin][1] <  new_lp:
                        new_ht, new_ct, new_attn_ctx, _ = model.decoder.lstm_step(embedded_o, st[1:3], context)
                        #x = model.decoder.calc_attentive_x(torch.cat([new_attn_ctx, embedded_o], dim=1))
                        new_st = [embedded_o, new_ht, new_ct, new_attn_ctx]
                        new_hyplist[argmin] = (seq+[o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                    else:
                        break
                else:
                    new_ht, new_ct, new_attn_ctx, _ = model.decoder.lstm_step(embedded_o, st[1:3], context)
                    #x = model.decoder.calc_attentive_x(torch.cat([new_attn_ctx, embedded_o], dim=1))
                    new_st = [embedded_o, new_ht, new_ct, new_attn_ctx]
                    new_hyplist.append((seq+[o], new_lp, new_st))
                    if len(new_hyplist) == beam:
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
        hyplist = new_hyplist
    
    if len(comp_hyplist):
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
        return maxhyps, best_state[1]
    else:
        return [([], 0)], None


def main():
    parser = argparse.ArgumentParser('predict')
    #utils.predict_opt(parser)
    #pred_args = parser.parse_args()
    
    # load default args
    test_file = '../data/squad-v1/sample.json'
    hps_file = '../exp/pretrain/hps.pkl'
    model_file = '../exp/pretrain/saved_models/dev_loss_3.1747903708753915.pth'
    if not os.path.exists(test_file):
        raise RuntimeError('No test file to read')
    if not os.path.exists(hps_file):
        raise RuntimeError('No default arguments file to load')
    with open(hps_file, 'rb') as f:
        hps = pickle.load(f)
    
    print('begin to load vocab')
    embedding_dict_file = os.path.join(os.path.dirname(hps.word_count_path), 'emb_dict.pkl')
    vocab = Vocab(hps.word_count_path, hps.glove_path, hps.embedding_dim, embedding_dict_file)
    print('begin to load pretrain model')
    model = torch.load(model_file)
    model.eval()
    #model = PointerNet(hps, vocab.emb_mat)
    #model = model.cuda()
    
    #reload_model(pred_args.load_path, pred_args.load_prefix, encoder, decoder)
    #print('begin to beam search')
    predict(test_file, vocab, model, hps)
    


if __name__ == '__main__':
    main()

