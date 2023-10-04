# import required libraries

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import random
import time

from io import open
import unicodedata

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

plt.switch_backend('agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set seed value
seed_value = 42

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
random.seed(seed_value)

# set start and end of the sequence
SOS_token = 0
EOS_token = 1


# implement language
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        # continuing count from index 2 after 'EOS' index 1
        self.word_index = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.word_index
            self.word2count[word] = 1
            self.index2word[self.word_index] = word
            self.word_index += 1
        else:
            self.word2count[word] += 1


# language pair data reading and preprocessing
# converting unicode into plain ASCII,
# acknowledge the brain behind the following script (https://stackoverflow.com/a/518232/2809427)

def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# case fold, trim whitespace from both sides, and remove non-letter characters
def normalizeString(s):
    # case fold, and trim
    s = unicode2ascii(s.lower().strip())
    # remove non-letter
    s = re.sub(r"([.!?])", r" \1", s)
    return s.strip()


def readLangs(lang_1, lang_2, reverse=False):
    print("Reading lines...")
    
    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang_1, lang_2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs of lang(l_1, l_2) and normalize
    pairs = [[normalizeString(s) for s in line.split('\t')] for line in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang_2)
        output_lang = Lang(lang_1)
    else:
        input_lang = Lang(lang_1)
        output_lang = Lang(lang_2)

    return input_lang, output_lang, pairs


# set maximum length for the sentence in both languages
MAX_LENGTH = 10

english_prefixes = (
    "i am ", "i m ",
    "he is ", "he s ",
    "she is ", "she s ",
    "you are ", "you re ",
    "we are ", "we re ",
    "they are ", "they re "
)


# Note: pair([fra, eng]) p[1].startswith(english_prefixes)
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH \
        and len(p[1].split(' ')) < MAX_LENGTH \
        and p[1].startswith(english_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang_1, lang_2, reserve=False):
    input_lang, output_lang, pairs = readLangs(lang_1, lang_2, reserve)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.word_index)
    print(output_lang.name, output_lang.word_index)

    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
# print(random.choice(pairs))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # initialize an embedding layer for converting input indexes to dense vectors
        self.embedding = nn.Embedding(input_size, hidden_size)
        # create a Gated Recurrent Unit (GRU) layer with the specified hidden size
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        ##
        # Autoencoders that include dropout are often called "denoising autoencoders"
        # because they use dropout to randomly corrupt the input,
        # with the goal of producing a network that is more robust to noise.
        ##

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        # pass the embedded vector through the GRU layer
        # 'output' contains the GRU outputs for each time step in the sequence
        # 'hidden' is the final hidden state of the GRU after processing the sequence
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_output, encoder_hidden, target_tensor=None):
        batch_size = encoder_output.size(0)
        # start of the sequence token
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output) # Collect the decoder outputs for each time step

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without Teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach() # detach from history as input

        # Concatenate decoder outputs along the sequence length
        # Apply log softmax to compute token probabilities
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return 'None' for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


#   Attention model building
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach() # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permuts(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


# return a list of indexes corresponding to the words in the sentence using the word2index mapping
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

# create input and target tensors for the entire dataset of language pairs
def tensorsFromPair(pair):
    input_tensor =  tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData("eng", "fra", True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

