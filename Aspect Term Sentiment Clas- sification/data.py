import pandas as pd
import time
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from transformers import BertTokenizer
from transformers import BertModel

train_df = pd.read_csv('./data/restaurants_train.csv')
test_df = pd.read_csv('./data/restaurants_test.csv')


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)