# -*- coding: utf-8 -*-
import argparse
import logging
import os

import numpy as np
import random
import torch
from config.grid_args import parse_args_grid

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(args):
    # args.train = 1
    # args.train_and_eval = 1
    # args.partial_load = 1
    # args.load_pretrained_model = 1
    # args.evaluate = 1

    print(args, end='\n\n')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.g)

    from runners.runner2 import Runner
    runner = Runner(args)
    if args.train:
        runner.train()
    if args.evaluate:
        runner.eval(args.dec)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = parse_args_grid()
    torch.autograd.set_detect_anomaly(True)
    main(args)
