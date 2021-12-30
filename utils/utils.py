import six
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from bpemb import BPEmb


class ByteTextEncoder:
    def __init__(self, num_reserved_ids=3):
        self._num_reserved_ids = num_reserved_ids
        if num_reserved_ids == 3:
            self.reserved_tokens = ['<PAD>', '<SOS>', '<EOS>']
        elif num_reserved_ids == 4:
            self.reserved_tokens = ['<PAD>', '<SOS>', '<EOS>', '<MASK>']
        else:
            self.reserved_tokens = ['<PAD>', '<SOS>', '<EOS>']
        self.reserved_ids = {token: id for id, token in enumerate(self.reserved_tokens)}

    def encode(self, s):
        return [c + self._num_reserved_ids for c in s.encode('utf-8')]

    def decode(self, ids):
        rcnt = self._num_reserved_ids
        return ''.join([six.int2byte(id - rcnt).decode("utf-8", "replace") if id >= rcnt else self.reserved_tokens[id] for id in ids])


class WordEncoder:
    def __init__(self, num_reserved_ids=3, vocab=None):
        self._num_reserved_ids = num_reserved_ids
        if num_reserved_ids == 3:
            self.reserved_tokens = ['<PAD>', '<SOS>', '<EOS>']
        elif num_reserved_ids == 4:
            self.reserved_tokens = ['<PAD>', '<SOS>', '<EOS>', '<MASK>']
        else:
            self.reserved_tokens = ['<PAD>', '<SOS>', '<EOS>']
        self.reserved_ids = {token: id for id, token in enumerate(self.reserved_tokens)}

        self.vocab = vocab
        self.id2vocab = {i: w for w, i in vocab.items()}

    def encode(self, s):
        return [self.vocab[w] + self._num_reserved_ids for w in word_tokenize(s) if w in self.vocab]

    def decode(self, ids):
        rcnt = self._num_reserved_ids
        return ' '.join([self.id2vocab[i - rcnt] if i >= rcnt else self.reserved_tokens[i] for i in ids])


class CharEncoder:
    def __init__(self, num_reserved_ids=3, vocab=None):
        self._num_reserved_ids = num_reserved_ids
        if num_reserved_ids == 3:
            self.reserved_tokens = ['<PAD>', '<SOS>', '<EOS>']
        elif num_reserved_ids == 4:
            self.reserved_tokens = ['<PAD>', '<SOS>', '<EOS>', '<MASK>']
        else:
            self.reserved_tokens = ['<PAD>', '<SOS>', '<EOS>']
        self.reserved_ids = {token: id for id, token in enumerate(self.reserved_tokens)}

        self.vocab = vocab
        self.id2vocab = {i: w for w, i in vocab.items()}

    def encode(self, s):
        return [self.vocab[c] + self._num_reserved_ids for c in s]  # char

    def decode(self, ids):
        rcnt = self._num_reserved_ids
        return ''.join([self.id2vocab[i - rcnt] if i >= rcnt else self.reserved_tokens[i] for i in ids])  # char




class BytePairEncoder:
    def __init__(self, num_reserved_ids=3, vocab=None):
        if num_reserved_ids == 3:
            self.reserved_tokens = ['<PAD>', '<SOS>', '<EOS>']
        elif num_reserved_ids == 4:
            self.reserved_tokens = ['<PAD>', '<SOS>', '<EOS>', '<MASK>']
        self.reserved_ids = {token: id for id, token in enumerate(self.reserved_tokens)}

        self.vocab = {w: i + num_reserved_ids for w, i in vocab.items()}
        self.vocab.update(self.reserved_ids)
        self.id2vocab = {i: w for w, i in self.vocab.items()}
        # sentence = "load English BPEmb model with 1000 vocabulary size and 300-dimensional embeddings"
        # sub = bpemb_en.encode(sentence)
        # sub_id = bpemb_en.encode_ids(sentence)
        # print(bpemb_en.emb.vocab)
        # emb = bpemb_en.vectors[sub_id]

    def encode(self, s):
        if type(s) == str:
            return [sub_id for sub_id in bpe.encode_ids(s)]
        elif type(s) == list:
            return [self.vocab[sub_id] for sub_id in s]

    def decode(self, ids):
        return ''.join([self.id2vocab[int(sub_id)] for sub_id in ids]).replace('▁', ' ').strip()
        # d = bpe.decode_ids([sub_id - self._num_reserved_ids for sub_id in ids]) 
        # return d if len(d) > 0 else ''


class BpeTextEncoder:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.id2w = [w.strip() for w in f.readlines()]
        self.w2id = {w: id for id, w in enumerate(self.id2w)}
        self.id2w.append('<SOS>')
        self.w2id['<SOS>'] = len(self.id2w) - 1
        self.reserved_ids = {'<PAD>': 0, '<EOS>': 1, '<SOS>': len(self.id2w) - 1}
        # a = ''.join(self.id2w)
        # import re
        # print(re.findall('\d+',a)) #['50', '20', '6', '0004', '0', '8', '2', '1', '00', '3', '5', '7', '9', '5', '6', '1', '3', '9', '2', '8', '19', '4', '0', '7']

    def encode(self, s):
        return [self.w2id[w] for w in s.split()]

    def decode(self, ids):
        s = ' '.join([self.id2w[id] for id in ids])
        # replace_list = [
        #     ('@@ ', ''), (' &apos;', '\'')
        # ]
        # for rep in replace_list:
        #     s = s.replace(rep[0], rep[1])
        return s


class Lexicon():
    def __init__(self, lexicon_file, phoneset_list, num_reserved_ids=3):
        self.lexicon = {}
        if num_reserved_ids == 3:
            self.reserved_tokens = ['<PAD>', '<SOS>', '<EOS>']
        elif num_reserved_ids == 4:
            self.reserved_tokens = ['<PAD>', '<SOS>', '<EOS>', '<MASK>']
        else:
            self.reserved_tokens = ['<PAD>', '<SOS>', '<EOS>']

        with open(phoneset_list, 'r') as f:
            self.phoneset = [l.strip() for l in f.readlines()]
        self.phoneset = self.reserved_tokens + self.phoneset

        self.phone2id = {p: id for id, p in enumerate(self.phoneset)}

        with open(lexicon_file, 'r') as f:
            for l in f.readlines():
                cur = l.split()
                self.lexicon[cur[0]] = [self.phone2id[p] for p in cur[1:]]

    def encode(self, s):
        res = []
        for w in s.split():
            res += self.lexicon[w.upper()]
        return res


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""

    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""

    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n