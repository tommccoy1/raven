import torch
import torch.nn.functional as F

import numpy as np

from fairseq.models.transformer_lm import TransformerLanguageModel
import argparse
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, default=20, help="Number of tokens to generate for each prompt")
parser.add_argument("--k", type=int, default=None, help="For top-k sampling: Only sample from the k most probable words")
parser.add_argument("--p", type=float, default=None, help="For top-p sampling (aka nucleus sampling): Only sample from the top p portion of the probability mass")
parser.add_argument("--temperature", type=float, default=None, help="temperature of 1.0 has no effect, lower tends toward greedy sampling")
parser.add_argument("--prompt_file", type=str, default=None, help="File of prompts, 1 prompt per line.")
parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to the model checkpoint")
parser.add_argument("--seed", default=7, type=int, help="Random seed")
args = parser.parse_args()

# These are all properties of the models we are using;
# we have hard-coded these as they are intended for one
# specific model
args.task = "language_modeling"
args.data = "../data/fairseq_data/"
args.output_dictionary_size = -1
args.arch = "transformer_lm_wiki103"
args.decoder_layers = 16
args.decoder_attention_heads = 8
args.dropout = 0.0
args.attention_dropout = 0.0
args.activation_dropout = 0.0
args.adaptive_softmax_dropout = 0.0
args.activation_fn = "relu"
args.decoder_normalize_before = True

args.relu_dropout = 0.0
args.adaptive_input = True
args.tie_adaptive_weights = True
args.adaptive_input_cutoff = '20000,60000'
args.adaptive_softmax_cutoff = '20000,60000'
args.no_decoder_final_norm = True
args.tie_adaptive_proj = True
args.decoder_embed_dim = 1024
args.decoder_input_dim = 1024
args.decoder_output_dim = 1024
args.adaptive_input_factor = 4
args.adaptive_softmax_factor = 4
args.decoder_ffn_embed_dim = 4096
args.decoder_layers_to_keep = None
args.character_embeddings = False
args.quant_noise_pq = 0
args.quant_noise_pq_block_size = 8
args.quant_noise_scalar = 0
args.decoder_layerdrop = 0
args.no_scale_embedding = False
args.layernorm_embedding = False
args.share_decoder_input_output_embed = False
args.no_token_positional_embeddings = False
args.decoder_learned_pos = False


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

set_seed(args)

# Set up task and model
task = tasks.setup_task(args)

model = task.build_model(args)
#model.load_state_dict(torch.load(args.model_name_or_path)["model"])
model.load_state_dict(torch.load(args.model_name_or_path, map_location=torch.device('cpu'))["model"])
model.eval()

# Set up tokenization dictionaries
trg_dict = task.target_dictionary

index2word = {}
for key in trg_dict.indices:
    index2word[trg_dict.indices[key]] = key

index2word[trg_dict.eos()] = "<eos>"
index2word[trg_dict.unk()] = "<unk>"


# Convert a prompt string into a tensor
# of token indices
def encode_prompt(prompt):
    encoding = []
    words = prompt.strip().split()
    for word in words:
        if word == "<eos>":
            encoding.append(trg_dict.eos())
        elif word not in trg_dict:
            encoding.append(trg_dict.unk())
        else:
            encoding.append(trg_dict.indices[word])

    return torch.LongTensor([encoding])


# Based in part on this snippet: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def generate(prompt, gen_length, topk=None, topp=None, temp=None):
    generation = []
    for gl in range(gen_length):
        print(gl)
        prompt_encoding = encode_prompt(prompt)
        output, _ = model(prompt_encoding)

        probs = model.decoder.adaptive_softmax.get_log_prob(output, target=None)[0][-1]
        probs = torch.exp(probs)

        if temp is not None:
            probs = torch.pow(probs, 1.0 / temp)
            probs = probs / probs.sum()

        if topk is not None:
            indices_to_remove = probs < torch.topk(probs, topk)[0][..., -1, None]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()

        if topp is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > topp

            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()


        index = torch.multinomial(probs, 1).item()
        word = index2word[index]

        generation.append(word)
        prompt = prompt + " " + word

    return " ".join(generation)


prompt_list = []
prompt_file = open(args.prompt_file, "r")

for line in prompt_file:
    prompt_list.append(line.strip())

output_filename = "_".join([str(x) for x in ["transformer", args.prompt_file.split("/")[-1], "k" + str(args.k), "p" + str(args.p), "temp" + str(args.temperature)]]) + ".generated"

fo = open(output_filename, "w")

for prompt in prompt_list:
    fo.write(generate(prompt, args.length, topk=args.k, topp=args.p, temp=args.temperature) + "\n")



