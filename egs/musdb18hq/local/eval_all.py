#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import musdb

from utils.utils import set_seed
from driver import EvaluaterBase as Evaluater

parser = argparse.ArgumentParser(description="Evaluation of D3Net")

parser.add_argument('--musdb18hq_root', type=str, default=None, help='Path to MUSDB18-HQ')
parser.add_argument('--estimated_musdb18hq_root', type=str, default=None, help='Path to estimated MUSDB18-HQ')
parser.add_argument('--sr', type=int, default=10, help='Sampling rate')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)

    loader = {}
    loader['mus'] = musdb.DB(root=args.musdb18hq_root, subsets="test", is_wav=True)
    loader['est'] = musdb.DB(root=args.estimated_musdb18hq_root, subsets="test", is_wav=True)
    
    evaluater = Evaluater(loader, args)
    evaluater.run()
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
