
import math
import argparse
from utils import batchify, repackage_hidden
import logging

import torch
import torch.nn as nn

from dictionary_corpus import Dictionary, Corpus, tokenize

parser = argparse.ArgumentParser(description='Generate text from LSTM language model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--topk', type=int, default=None, help='k to use for top-k generation')
parser.add_argument('--topp', type=float, default=None, help='p to use for top-p generation')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature to use for generation')
parser.add_argument('--prompt_file', type=str, default=None, help='file of prompts to generate from')
parser.add_argument('--length', type=int, default=None, help='sequence length to generate')
args = parser.parse_args()

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logging.info(args)

def get_batch(source, i, seq_length, min_seq_len=True):
    if min_seq_len:
        seq_len = min(seq_length, len(source) - 1 - i)
    else:
        seq_len = seq_length
    data = source[i:i+seq_len]
    # predict the sequences shifted by one word
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def tokenize_string(dictionary, words):
    ntokens = len(words)
    ids = torch.LongTensor(ntokens)

    token = 0

    for word in words:
        if word in dictionary.word2idx:
            ids[token] = dictionary.word2idx[word]
        else:
            ids[token] = dictionary.word2idx["<unk>"]
        token += 1

    return ids

def generate(model, data_source, length, topk=None, topp=None, temperature=1.0):
    model.eval()

    ntokens = len(dictionary)

    hidden = model.init_hidden(1) # 1 is the batch size
    unk_idx = dictionary.word2idx["<unk>"]

    if args.cuda:
        out_type = torch.cuda.LongTensor()
    else:
        out_type = torch.LongTensor()

    with torch.no_grad():
        data, targets = get_batch(data_source, 0, data_source.shape[0], min_seq_len=False)
        output, hidden = model(data, hidden)

        generated = []
        for i in range(length):
            output_flat = output.view(-1, ntokens)
            logits = output_flat[-1] / temperature

            if topk is not None:
                indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            if topp is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nn.Softmax()(sorted_logits), dim=-1)

                sorted_indices_to_remove = cumulative_probs > topp

                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]

                logits[indices_to_remove] = -float('Inf')

            probs = nn.Softmax()(logits)


            probs = torch.distributions.Categorical(probs)
            next_word = dictionary.idx2word[probs.sample()]
            generated.append(next_word)

            inp = batchify(tokenize_string(dictionary, [next_word]), 1, args.cuda)

            output, hidden = model(inp, hidden)

    return generated


dictionary = Dictionary(args.data)

# Load the best saved model.
with open(args.checkpoint, 'rb') as f:
    logging.info("Loading the model")
    if args.cuda:
        model = torch.load(f)
    else:
        # to convert model trained on cuda to cpu model
        model = torch.load(f, map_location=lambda storage, loc: storage)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model))
15/0


fi_prompt = open(args.prompt_file, "r")
fo = open("/".join(args.prompt_file.split("/")[:-1]) + "/lstm_" + args.prompt_file.split("/")[-1] + "_topk_" + str(args.topk) + "_topp_" + str(args.topp) + "_temp_" + str(args.temperature) + "_length_" + str(args.length) + ".generated", "w")
for index, line in enumerate(fi_prompt):
    logging.info(str(index))
    prompt = tokenize_string(dictionary, line.strip().split())
    prompt_data = batchify(prompt, 1, args.cuda)
    ntokens = len(dictionary)

    generation = generate(model, prompt_data, args.length, topk=args.topk, topp=args.topp, temperature=args.temperature)
    fo.write(" ".join(generation) + "\n")

