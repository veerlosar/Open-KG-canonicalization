#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import torch
import logging
import pandas as pd

from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Set
from collections import defaultdict as ddict

from helper import invertDict

logging.basicConfig(level=logging.INFO)

class BuildDataForCanonicalization(Dataset):
    def __init__(self, data_dir):
        if data_dir is None or not os.path.exists(data_dir): raise FileNotFoundError
        self.data_dir = data_dir
        self.ent2id = dict(self.read_file(os.path.join(self.data_dir, 'ent2id.txt')))
        self.id2ent = dict([(idx, entity) for entity, idx in self.ent2id.items()])
        self.rel2id = dict(self.read_file(os.path.join(self.data_dir, 'rel2id.txt')))
        self.id2rel = dict([(idx, rel) for rel, idx in self.rel2id.items()])
        self.triples = self.read_file(os.path.join(self.data_dir, 'triples.txt'))
        '''Added links between triples and rels'''
        # Thoughts: removing duplicates in the triples.
        # This works for relations, but might not for entities.
        # Duplicates may be there on purpose if, e.g. Apple and apple match to two
        # different entities in Wiki in exactly the same triple
        self.triples = list({tuple(t) for t in self.triples})
        self.triple2relid = {trip: self.rel2id[trip[1]] for trip in self.triples}
        '''Leaving only one triple per rel for dimensionality's sake.
        Also, it should not be a problem for any model to cluster exactly the same relations
        together'''
        self.relid2untriple = {idx: trip for trip, idx in self.triple2relid.items()}
        self.untriple2relid = {trip: idx for idx, trip in self.relid2untriple.items()}
        ''''''
        entity_side_info = self.read_file(os.path.join(self.data_dir, 'ent_side_info.txt'))
        self.ent_side_info = list(map(lambda z: (self.ent2id[z[0]], self.ent2id[z[1]], float(z[2])), entity_side_info))
        relation_side_info = self.read_file(os.path.join(self.data_dir, 'rel_side_info.txt'))
        self.rel_side_info = list(map(lambda z: (self.rel2id[z[0]], self.rel2id[z[1]], float(z[2])), relation_side_info))
        # Golden clusters dataset for entities
        self.ent2truelinks = self.read_gold_clust(os.path.join(self.data_dir, 'gold_npclust.txt'))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, item):
        if torch.is_tensor(item): item = item.tolist()
        head, rel, tail = self.triples[item]   # (h,r,t)
        head_idx = self.ent2id[head]
        tail_idx = self.ent2id[tail]
        rel_idx = self.rel2id[rel]
        '''Triple is returned instead of a single rel now'''
        # curr_triple = [head_idx, rel_idx, tail_idx]
        curr_triple = [head_idx, rel_idx, tail_idx]
        ''''''
        ent_side_info = self.ent_side_info[item % len(self.ent_side_info)]
        rel_side_info = self.rel_side_info[item % len(self.rel_side_info)]
        return curr_triple, ent_side_info, rel_side_info

    @staticmethod
    def read_file(f_name: str) -> List[Tuple[str, str, str]]:
        if f_name is None or not os.path.exists(f_name): return FileNotFoundError
        df = pd.read_csv(f_name, sep='\t')
        entries = list(df.to_records(index=True))
        logging.info('\nFile: {0}\nNumber of entries READ: {1}'.format(f_name, len(entries)))
        return entries

    @staticmethod
    def read_gold_clust(f_name: str)->Dict[str, Set[str]]:
        if f_name is None or not os.path.exists(f_name): return None
        clustid2entities = ddict(set)
        with open(f_name, 'r') as f:
            for entry in f:
                entries = entry.strip().split('\t')
                entities = entries[2:]
                if entries[0] in clustid2entities: raise KeyError
                clustid2entities[entries[0]] = entities
        return invertDict(clustid2entities)
