def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')
    group.add_argument('-word_vec_size', type=int, default=300,
                       help='Word embedding size for src and tgt.')

    # Encoder-Deocder Options
    group = parser.add_argument_group('Model-Encoder-Decoder')

    group.add_argument('-enc_layers', type=int, default=2,
                       help='Number of layers in the encoder')
    group.add_argument('-dec_layers', type=int, default=1,
                       help='Number of layers in the decoder')
    group.add_argument('-rnn_size', type=int, default=100,
                       help='Size of rnn hidden states')
    group.add_argument('-use_attention', default=False, action='store_true',
                       help='Whether to use attention mechnism')
    group.add_argument('-use_copy', default=False, action='store_true',
                       help='Whether to use copy mechnism')

    ## Genenerator and loss options.
    #group.add_argument('-copy_attn', action="store_true",
    #                   help='Train copy attention layer.')
    #group.add_argument('-copy_attn_force', action="store_true",
    #                   help='When available, train to copy.')
    #group.add_argument('-reuse_copy_attn', action="store_true",
    #                   help="Reuse standard attention for copy")
    #group.add_argument('-copy_loss_by_seqlength', action="store_true",
    #                   help="Divide copy loss by length of sequence")
    #group.add_argument('-coverage_attn', action="store_true",
    #                   help='Train a coverage attention layer.')
    #group.add_argument('-lambda_coverage', type=float, default=1,
    #                   help='Lambda value for coverage.')


def data_opts(parser):
    # Data options
    group = parser.add_argument_group('Data')
    
    group.add_argument('-data_dir', required=True,
                       help="Directory to the store all necessary data")
    group.add_argument('-train_file', default="train_raw.json",
                       help="Name of the training data file")
    group.add_argument('-valid_file', default="dev_raw.json",
                       help="Name of the valid data file")

    # Dictionary options, for text corpus
    group = parser.add_argument_group('Vocab')
    group.add_argument('-word_count_file', default="word_counter.json",
                       help="""Name of the word counter file, 
                       which store the word frequency on training data""")
    group.add_argument('-glove_path', default="~/data/glove/glove.840B.300d.txt",
                       help="Path of the glove pretrain vector file")
    group.add_argument('-max_vocab_size', type=int, default=50000,
                       help="""Size of the vocabulary, for the most common order in word_count_file,
                       the program to check whether each word exists in glove pretrain file,
                       if exists, add to emb_mat, when the length of emb_mat equals to max_vocab_size,
                       the construction of vocab finish""")
    group.add_argument('-update_embedding', default=False, action='store_true',
                       help="To fix pretrain word embedding or update its weights")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add_argument('-src_seq_length', type=int, default=65,
                       help="Maximum source sequence length")
    group.add_argument('-tgt_seq_length', type=int, default=65,
                       help="Maximum target sequence length to keep.")

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add_argument('-shuffle', default=False, action='store_true',
                       help="Shuffle data")
    group.add_argument('-seed', type=int, default=3435,
                       help="Random seed")


def train_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('General')
    group.add_argument('-save_model', default=False, action='store_true',
                       help="""Whether to save model according to best valid loss""")

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add_argument('-batch_size', type=int, default=128,
                       help='Maximum batch size for training')
    group.add_argument('-valid_steps', type=int, default=10000,
                       help='Perfom validation every X steps')
    group.add_argument('-valid_batch_size', type=int, default=32,
                       help='Maximum batch size for validation')
    group.add_argument('-train_steps', type=int, default=100000,
                       help='Number of training steps')
    group.add_argument('-epochs', type=int, default=0,
                       help='Deprecated epochs see train_steps')
    group.add_argument('-max_grad_norm', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to
                       max_grad_norm""")
    group.add_argument('-dropout', type=float, default=0.3,
                       help="Dropout probability; applied in LSTM stacks.")

    group = parser.add_argument_group('Logging')
    group.add_argument('-report_every', type=int, default=50,
                       help="Print stats at this interval.")
    group.add_argument('-exp_dir', type=str, default="",
                       help="Name of the experiment for logging.")


def translate_opts(parser):
    """ Translation / inference options """
    group = parser.add_argument_group('Model')
    group.add_argument('-model', required=True,
                       help='Path to model .pt file')

    group = parser.add_argument_group('Data')
    group.add_argument('-test_file', required=True,
                       help="""""")
    group.add_argument('-output', default='pred.txt',
                       help="""Path to output the predictions (each line will
                       be the decoded sequence""")
    group.add_argument('-report_bleu', action='store_true',
                       help="""Report bleu score after translation,
                       call tools/multi-bleu.perl on command line""")

    group = parser.add_argument_group('Beam')
    group.add_argument('-beam_size', type=int, default=5,
                       help='Beam size')
    group.add_argument('-max_length', type=int, default=100,
                       help='Maximum prediction length.')

    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    #group.add_argument('-stepwise_penalty', action='store_true',
    #                   help="""Apply penalty at every decoding step.
    #                   Helpful for summary penalty.""")
    #group.add_argument('-length_penalty', default='none',
    #                   choices=['none', 'wu', 'avg'],
    #                   help="""Length Penalty to use.""")
    #group.add_argument('-coverage_penalty', default='none',
    #                   choices=['none', 'wu', 'summary'],
    #                   help="""Coverage Penalty to use.""")
    #group.add_argument('-alpha', type=float, default=0.,
    #                   help="""Google NMT length penalty parameter
    #                    (higher = longer generation)""")
    #group.add_argument('-beta', type=float, default=-0.,
    #                   help="""Coverage penalty parameter""")
    #group.add_argument('-block_ngram_repeat', type=int, default=0,
    #                   help='Block repetition of ngrams during decoding.')
    #group.add_argument('-ignore_when_blocking', nargs='+', type=str,
    #                   default=[],
    #                   help="""Ignore these strings when blocking repeats.
    #                   You want to block sentence delimiters.""")
    #group.add_argument('-replace_unk', action="store_true",
    #                   help="""Replace the generated UNK tokens with the
    #                   source token that had highest attention weight. If
    #                   phrase_table is provided, it will lookup the
    #                   identified source token and give the corresponding
    #                   target token. If it is not provided(or the identified
    #                   source token does not exist in the table) then it
    #                   will copy the source token""")

    group = parser.add_argument_group('Logging')
    group.add_argument('-verbose', action="store_true",
                       help='Print scores and predictions for each sentence')
    group.add_argument('-attn_debug', action="store_true",
                       help='Print best attn for each word')
    group.add_argument('-dump_beam', type=str, default="",
                       help='File to dump beam information to.')
    group.add_argument('-n_best', type=int, default=1,
                       help="""If verbose is set, will output the n_best
                       decoded sentences""")

    group = parser.add_argument_group('Efficiency')
    group.add_argument('-batch_size', type=int, default=30,
                       help='Batch size')

