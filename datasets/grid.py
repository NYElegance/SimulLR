import sys

sys.path.append('../')
import json
from random import random, randint
import numpy as np, cv2 as cv
from torch.utils.data import Dataset
from utils.utils import WordEncoder, CharEncoder, BytePairEncoder


class GridDataset(Dataset):
    def __init__(self, data_path, args, rate=None):
        self.train = False
        if 'train' in data_path and args.train:
            self.train = True

        with open(data_path, encoding='utf-8') as f:
            self.data = json.load(f)

        self.vocab = args.vocab if args.use_word else None

        self.text_max_length = args.text_max_length  # char 41, word 8
        self.video_max_length = args.video_max_length  # 75
        if args.text_level == 'word':
            self.text_encoder = WordEncoder(3, args.vocab)
        elif args.text_level == 'char':
            self.text_encoder = CharEncoder(3, args.vocab)
        elif args.text_level == 'subword':
            self.text_encoder = BytePairEncoder(3, args.vocab)

    def __getitem__(self, index):
        npz_path, text = self.data[index]

        tmp = np.load(npz_path)

        x = tmp['vid_array']  # (v_l:75, 60, 100)
        x_mask = tmp['vid_mask']  # (v_l:75,) All True
        if not self.train:
            return {'x': x.astype(np.float32),
                    'x_mask': x_mask,
                    'y': text}

        if random() > 0.6:
            x = x[:, :, ::-1]
        if random() > 0.8:
            l, h, w = x.shape
            new = np.zeros(x.shape)
            dh = randint(-3, 3)
            dw = randint(-5, 5)
            new[:, max(0, -dh):h - dh, max(0, -dw):w - dw] = x[:, max(0, dh):h + dh, max(0, dw):w + dw]  # crop
            x = new

        text_ids = self.text_encoder.encode(text)
        if len(text_ids) > self.text_max_length:
            text_ids = text_ids[:self.text_max_length - 1] + [self.text_encoder.reserved_ids['<EOS>']]
        elif len(text_ids) == self.text_max_length:
            text_ids[-1] = self.text_encoder.reserved_ids['<EOS>']
        elif len(text_ids) < self.text_max_length:
            text_ids = text_ids + \
                       [self.text_encoder.reserved_ids['<EOS>']] + \
                       [self.text_encoder.reserved_ids['<PAD>']] * (self.text_max_length - len(text_ids) - 1)

        text_ids = np.asarray(text_ids)
        text_mask = text_ids != 0
        return {
            'x': x.astype(np.float32),
            'x_mask': x_mask,
            'y': text_ids,
            'y_mask': text_mask
        }

    def __len__(self):
        return len(self.data)

