import torch
from torchtext.data import get_tokenizer
from torch.utils.data import TensorDataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.functional import to_tensor, truncate

import pandas as pd
from pandas import DataFrame, read_csv
import numpy as np

tokenizer = get_tokenizer('basic_english')

def readCleanData(filename):
    data = pd.read_csv(filename)
    # cast data['Content'] to string
    data['Content'] = data['Content'].astype(str)
    data['AuthorID'] = data['AuthorID'].astype(int)
    return data

data = readCleanData('cleanedData.csv')

def iter_tokens(df):
    txt = df['Content']
    tokenized_txt = [tokenizer(t) for t in txt]
    return iter(tokenized_txt)

def getVocab(df, max_tokens):
    vocab = build_vocab_from_iterator(iter_tokens(df), max_tokens=max_tokens, specials=['<pad>' ,'<unk>', '<start>'])
    return vocab