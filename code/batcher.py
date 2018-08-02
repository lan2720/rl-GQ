# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to process data into batches"""

from multiprocessing import Queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import data
import spacy
import ujson as json

# used for spacy
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))


def word_tokenize(sent):
    doc = nlp(sent)
    return list(filter(lambda i: i.strip() != '', [token.text for token in doc]))


class Example(object):
    """Class representing a train/val/test example for text summarization."""

    def __init__(self, paragraph, question, answer, answer_indices, vocab, hps):
        """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        Args:
            article: source text; a string. each token is separated by a single space.
            abstract_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
            vocab: Vocabulary object
            hps: hyperparameters
        """
        self.hps = hps

        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        paragraph_words = word_tokenize(paragraph)
        question_words = word_tokenize(question)
        answer_start_idx, answer_end_idx = answer_indices
        
        # Process the paragraph
        if len(paragraph_words) > hps.max_enc_steps:
            if answer_end_idx <= hps.max_enc_steps:
                paragraph_words = paragraph_words[:hps.max_enc_steps]
            else:
                answer_mid_idx = (answer_start_idx + answer_end_idx) // 2
                # assume len(answer_words) <= len(paragraph_words)
                paragraph_trunc_end = min(answer_mid_idx + hps.max_enc_steps//2, len(paragraph_words))
                paragraph_trunc_start = paragraph_trunc_end - hps.max_enc_steps + 1
                assert (paragraph_trunc_start <= answer_start_idx) and (paragraph_trunc_end >= answer_end_idx) 
                paragraph_words = paragraph_words[paragraph_trunc_start:paragraph_trunc_end]
                answer_start_idx -= paragraph_trunc_start
                answer_end_idx -= paragraph_trunc_start
        self.enc_len = len(paragraph_words) # store the length after truncation but before padding
        self.enc_input = [vocab.word2id(w) for w in paragraph_words] # list of word ids; OOVs are represented by the id for UNK token

        # Process the abstract
        question_ids = [vocab.word2id(w) for w in question_words] # list of word ids; OOVs are represented by the id for UNK token

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(question_ids, hps.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if hps.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
            self.enc_input_extend_vocab, self.paragraph_oovs = data.article2ids(paragraph_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            question_ids_extend_vocab = data.abstract2ids(question_words, vocab, self.paragraph_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_targ_seqs(question_ids_extend_vocab, hps.max_dec_steps, start_decoding, stop_decoding)

        # Store the original strings
        self.original_paragraph = paragraph
        self.original_question = question
        self.original_answer = answer #' '.join(paragraph_words[answer_start_idx:answer_end_idx])
        self.answer_start_idx = answer_start_idx
        self.answer_end_idx = answer_end_idx
        #self.original_abstract_sents = abstract_sentences


    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).

        Args:
            sequence: List of ids (integers)
            max_len: integer
            start_id: integer
            stop_id: integer

        Returns:
            inp: sequence length <=max_len starting with start_id
            target: sequence same length as input, ending with stop_id only if there was no truncation
        """
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len: # truncate
            inp = inp[:max_len]
            target = target[:max_len] # no end_token
        else: # no truncation
            target.append(stop_id) # end token
        assert len(inp) == len(target)
        return inp, target


    def pad_decoder_inp_targ(self, max_len, pad_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)


    def pad_encoder_input(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if self.hps.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
    """Class representing a minibatch of train/val/test examples for text summarization."""

    def __init__(self, example_list, hps, vocab):
        """Turns the example_list into a Batch object.

        Args:
             example_list: List of Example objects
             hps: hyperparameters
             vocab: Vocabulary object
        """
        self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
        self.batch_size = len(example_list)
        self.init_answer_pos(example_list, hps)
        self.init_encoder_seq(example_list, hps) # initialize the input to the encoder
        self.init_decoder_seq(example_list, hps) # initialize the input and targets for the decoder
        self.store_orig_strings(example_list) # store the original strings


    def init_answer_pos(self, example_list, hps):
        self.ans_indices = np.zeros((self.batch_size, 2), dtype=np.int32)

        for i, ex in enumerate(example_list):
            self.ans_indices[i, 0] = ex.answer_start_idx
            self.ans_indices[i, 1] = ex.answer_end_idx

    def init_encoder_seq(self, example_list, hps):
        """Initializes the following:
                self.enc_batch:
                    numpy array of shape (batch_size, <=max_enc_steps) containing integer ids (all OOVs represented by UNK id), padded to length of longest sequence in the batch
                self.enc_lens:
                    numpy array of shape (batch_size) containing integers. The (truncated) length of each encoder input sequence (pre-padding).
                self.enc_padding_mask:
                    numpy array of shape (batch_size, <=max_enc_steps), containing 1s and 0s. 1s correspond to real tokens in enc_batch and target_batch; 0s correspond to padding.

            If hps.pointer_gen, additionally initializes the following:
                self.max_art_oovs:
                    maximum number of in-article OOVs in the batch
                self.art_oovs:
                    list of list of in-article OOVs (strings), for each example in the batch
                self.enc_batch_extend_vocab:
                    Same as self.enc_batch, but in-article OOVs are represented by their temporary article OOV number.
        """
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        #self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            #for j in xrange(ex.enc_len):
            #    self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if hps.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_para_oovs = max([len(ex.paragraph_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.para_oovs_batch = [ex.paragraph_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list, hps):
        """Initializes the following:
                self.dec_batch:
                    numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.
                self.target_batch:
                    numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.
                self.dec_padding_mask:
                    numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
                """
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
        self.dec_batch = np.zeros((self.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, hps.max_dec_steps), dtype=np.int32)
        #self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:] # ex.target has consider the oov word
            #for j in xrange(ex.dec_len):
            #    self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        """Store the original article and abstract strings in the Batch object"""
        self.original_paragraphs = [ex.original_paragraph for ex in example_list] # list of string
        self.original_questions = [ex.original_question for ex in example_list] # list of string
        self.original_answers = [ex.original_answer for ex in example_list] # list of string


class Batcher(object):
    BATCH_QUEUE_MAX = 10 # max number of batches the batch_queue can hold

    def __init__(self, data_path, vocab, hps, single_pass):
        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._single_pass = single_pass
        self._bucketing_cache_size = 3

        self.examples = json.load(open(data_path))
        self.data_size = len(self.examples)

        self._batch_queue = Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)


    def setup(self):
        self._finished_reading = False
        self._prepare_to_stop = False
        self._ready_to_stop = False
        self._stop = False

        assert (self._hps.batch_size*self._bucketing_cache_size) < self.data_size, 'batch_size is too large'

        self._example_q_thread = Thread(target=self.fill_example_queue)
        self._example_q_thread.daemon = True
        self._example_q_thread.start()
        
        self._batch_q_thread = Thread(target=self.fill_batch_queue)
        self._batch_q_thread.daemon = True
        self._batch_q_thread.start()


    def fill_example_queue(self):
        input_gen = text_generator(data.example_generator(self.examples, self._single_pass))

        while True:
            try:
                (paragraph, question, answer, answer_position) = next(input_gen) # read the next example from file. article and abstract are both strings.
            except StopIteration: # if there are no more examples:
                self._finished_reading = True
                print('reading data one round finish')
                break
            example = Example(paragraph, question, answer, answer_position, self._vocab, self._hps)
            self._example_queue.put(example)
            #print('put one in example queue: qsize %i' % self._example_queue.qsize())


    def fill_batch_queue(self):
        """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.

        In decode mode, makes batches that each contain a single example repeated.
        """
        while True:
            if self._prepare_to_stop:
                self._ready_to_stop = True
                break

            if self._hps.mode != 'decode':
                inputs = []
                for _ in range(self._hps.batch_size * self._bucketing_cache_size):
                    #print('example qsize:', self._example_queue.qsize())
                    if self._finished_reading and self._example_queue.qsize() == 0:
                        self._prepare_to_stop = True
                        break
                    inputs.append(self._example_queue.get())
                    #print('get one from example queue, current qsize:', self._example_queue.qsize())
                #print('inputs num:', len(inputs))
                inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True) # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self._hps.batch_size):
                    end_i = min(i+self._hps.batch_size, len(inputs))
                    batches.append(inputs[i:end_i])
                    #print('batch size:', len(batches[-1]))
                if not self._single_pass:
                    shuffle(batches) # in each batch, the example is sorted by enc_len
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._hps, self._vocab))
                    #print('batch queue put')

            else: # beam search decode mode
                ex = self._example_queue.get()
                b = [ex for _ in range(self._hps.batch_size)]
                self._batch_queue.put(Batch(b, self._hps, self._vocab))


    def next_batch(self):
        if self._ready_to_stop and self._batch_queue.qsize() == 0:
            self._stop = True
            raise StopIteration('no more batch')
        if self._batch_queue.qsize() == 0:
            print('current batch queue size: %i example queue size: %i' % (self._batch_queue.qsize(), self._example_queue.qsize()))
        batch = self._batch_queue.get() # get the next Batch
        return batch


#def newbatcher(vocab, hps, data_path, single_pass):
#    example_list = []
#    input_gen = text_generator(data.example_generator(data_path, single_pass))
#    while True:
#        try:
#            (paragraph, question, answer, answer_position) = next(input_gen) # read the next example from file. article and abstract are both strings.
#        except StopIteration: # if there are no more examples:
#            if single_pass:
#                print("single_pass mode is on, so we've finished reading dataset.")
#            break
#
#        example = Example(paragraph, question, answer, answer_position, vocab, hps) # Process into an Example.
#        example_list.append(example)
#        if len(example_list) == hps.batch_size:
#            example_list = sorted(example_list, key=lambda inp: inp.enc_len, reverse=True)
#            # NOTE: here all batch examples have been sorted according to the enc_len
#            yield Batch(example_list, hps, vocab)
#            example_list = []
#
#    # for single pass
#    if len(example_list) > 0:
#        example_list = sorted(example_list, key=lambda inp: inp.enc_len, reverse=True)
#        yield Batch(example_list, hps, vocab)


def text_generator(example_generator):
    """Generates article and abstract text from tf.Example.

    Args:
        example_generator: a generator of tf.Examples from file. See data.example_generator"""
    while True:
        e = next(example_generator) # e is a dict
        try:
            if not e['ifkeep']:
                continue
            paragraph_text = e['correct_sentence']
            question_text = e['question']
            answer_text = e['valid_answer'][0]
            answer_position = (e['ans_start_in_sent'], e['ans_end_in_sent'])
        except ValueError:
            print('Failed to get paragraph or question or answer position from example')
            continue
        yield (paragraph_text, question_text, answer_text, answer_position)


def id2sentence(inputs, vocab, extend_vocab):
    single_pass = True
    if isinstance(inputs, np.ndarray):
        if inputs.ndim > 2:
            raise Exception('Now inputs.ndim = %d, please note that inputs ndim must <= 2' % inputs.ndim)
        elif inputs.ndim == 2:
            single_pass = False
        inputs = inputs.tolist()
    if single_pass:
        inputs = [inputs]
    
    decoding = []
    for i, sent in enumerate(inputs):
        sent_dec = []
        for j in sent:
            try:
                w = vocab.id2word(j)
            except ValueError:
                try:
                    w = extend_vocab[i][j-vocab.size()]
                except:
                    raise Exception('Id not found in both vocab and extend vocab: %d' % j)
            sent_dec.append(w)
        decoding.append(sent_dec)
    if len(decoding) == 1:
        return decoding[0]
    else:
        return decoding
    
