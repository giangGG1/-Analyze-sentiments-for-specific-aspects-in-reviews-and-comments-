import pandas as pd
import time
import numpy as np
from sklearn.metrics import classification_report

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel


train_df = pd.read_csv('./data/restaurants_train.csv')
test_df = pd.read_csv('./data/restaurants_test.csv')