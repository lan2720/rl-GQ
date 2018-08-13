import os
import time
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import opts
import data
from batcher import Batcher
from decoder import Decoder
from seq2seq import Seq2Seq

def main():
    parser = argparse.ArgumentParser(
            description='train for basic seq2seq generator')
    opts.data_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()

    # Load vocab
    word_count_path = os.path.join(opt.data_dir, opt.word_count_file)
    print(vars(opt))

if __name__ == '__main__':
    main()
    

