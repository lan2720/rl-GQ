from batcher import Batcher, Example, Batch, text_generator, id2sentence
from data import Vocab
import data
import argparse
import time
import pickle

# ('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.'
# --data_path=/path/to/chunked/train_*
parser = argparse.ArgumentParser(description='arguments.')
parser.add_argument('--mode', type=str, default='train', choices=['eval', 'decode', 'train'],
                    help='the data path to load raw data')
parser.add_argument('--data_path', type=str, default='../data/squad-v1/train_raw.json',
                    help='the data path to load raw data')
parser.add_argument('--batch_size', type=int, default=16,
                    help='the num of examples in one batch')
parser.add_argument('--max_enc_steps', type=int, default=65,
                    help='the num of examples in one batch')
parser.add_argument('--max_dec_steps', type=int, default=65,
                    help='the num of examples in one batch')
parser.add_argument('--single_pass', type=bool, default=False,
                    help='whether pass the only one example each time') # only True when decoding
parser.add_argument('--pointer_gen', default=False, action='store_true',
                    help='whether to use pointer mechanism') # only True when decoding

wordcount_file = '/home/jiananwang/rl-QG/data/squad-v1/word_counter.json'
emb_file = '/home/jiananwang/data/glove/glove.840B.300d.txt'
#Vocab(wordcount_file, emb_file, 300)
hps = parser.parse_args()


vocab = Vocab(wordcount_file, emb_file, 300)
#batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=single_pass)
question = 'What is one of the ways that Special Operations is different from conventional methods?'
answer_indices=(27, 32)
paragraph='Special operations differ from conventional operations in degree of physical and political risk, operational techniques, mode of employment, independence from friendly support, and dependence on detailed operational intelligence and indigenous assets" (JP 1-02).'
answer = 'dependence on detailed operational intelligence'

def test_example():
    ex = Example(paragraph, question, answer, answer_indices, vocab, hps)
    print('enc len:', ex.enc_len)
    print('enc input:', ex.enc_input)
    print('enc input words:', ' '.join([vocab.id2word(i) for i in ex.enc_input]))
    print('dec len:', ex.dec_len)
    print('dec input:', ex.dec_input)
    print('dec input words:', ' '.join([vocab.id2word(i) for i in ex.dec_input]))
    print('dec target:', ex.target)
    print('dec target words:', ' '.join([vocab.id2word(i) for i in ex.target]))
    if hps.pointer_gen:
        print('enc input extend vocab:', ex.enc_input_extend_vocab)
        print('paragraph oov:', ex.paragraph_oovs)
    print('original paragraph:', ex.original_paragraph)
    print('original question:', ex.original_question)
    print('original answer:', ex.original_answer)
    print('ans start pos:', ex.answer_start_idx)
    print('ans end pos:', ex.answer_end_idx)

def test_batch():
    # python test_batcher.py --data_path=../data/squad-v1/dev_raw.json --pointer_gen
    input_gen = text_generator(data.example_generator(hps.data_path, hps.single_pass))
    example_list = []
    for _ in range(hps.batch_size):
        p, q, a, ap = next(input_gen)
        example_list.append(Example(p, q, a, ap, vocab, hps))
    batch = Batch(example_list, hps, vocab)
    print('batch answer pos:', batch.ans_indices)
    print('enc batch:', batch.enc_batch)
    print('enc batch words:', id2sentence(batch.enc_batch, vocab, batch.para_oovs_batch))
    print('enc len:', batch.enc_lens)
    if hps.pointer_gen:
        print('max para oovs:', batch.max_para_oovs)
        print('para oovs:', batch.para_oovs_batch)
        print('enc batch extend vocab:', batch.enc_batch_extend_vocab)
        print('enc batch extend vocab words:', id2sentence(batch.enc_batch_extend_vocab, vocab, batch.para_oovs_batch))
    print('dec batch:', batch.dec_batch)
    print('dec batch words:', id2sentence(batch.dec_batch, vocab, batch.para_oovs_batch))
    print('target batch:', batch.target_batch)
    print('tgt batch words:', id2sentence(batch.target_batch, vocab, batch.para_oovs_batch))
    print('origin para:', batch.original_paragraphs)
    print('origin question:', batch.original_questions)
    print('origin answer:', batch.original_answers)

def test_batcher():
    batcher = Batcher(hps.data_path, vocab, hps, hps.single_pass)
    #time.sleep(15)
    while True:
        start = time.time()
        batch = batcher.next_batch()
        print('elapse:', time.time()-start)
        pickle.dump(batch, open('one_batch.pkl', 'wb'))
        print('finish')
        break
        #print('batch answer pos:', batch.ans_indices)
        #print('enc batch:', batch.enc_batch)
        #print('enc batch words:', id2sentence(batch.enc_batch, vocab, batch.para_oovs_batch))
        #print('enc len:', batch.enc_lens)
        #if hps.pointer_gen:
        #    print('max para oovs:', batch.max_para_oovs)
        #    print('para oovs:', batch.para_oovs_batch)
        #    print('enc batch extend vocab:', batch.enc_batch_extend_vocab)
        #    print('enc batch extend vocab words:', id2sentence(batch.enc_batch_extend_vocab, vocab, batch.para_oovs_batch))
        #print('dec batch:', batch.dec_batch)
        #print('dec batch words:', id2sentence(batch.dec_batch, vocab, batch.para_oovs_batch))
        #print('target batch:', batch.target_batch)
        #print('tgt batch words:', id2sentence(batch.target_batch, vocab, batch.para_oovs_batch))
        #print('origin para:', batch.original_paragraphs)
        #print('origin question:', batch.original_questions)
        #print('origin answer:', batch.original_answers)
    
if __name__ == '__main__':
    test_batcher()
    #test_example()
