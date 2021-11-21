#!/usr/bin/env python3
# coding=utf-8

# This script is a modified version of 'run_generation.py' from
# the HuggingFace GitHub repo, available here:
# https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py
# Below is the license information from the original script.
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

import argparse
import logging
import ftfy

import numpy as np
import torch

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
}

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# Convert a list of prompts into batches
def batchify_prompts(prompt_list, batch_size):
    batches = []

    this_batch = []
    for prompt in prompt_list:
        this_batch.append(prompt)
        if len(this_batch) == batch_size:
            batches.append(this_batch[:])
            this_batch = []
    if len(this_batch) > 0:
        batches.append(this_batch)

    return batches

def pad_batch(batch, pad_idx):
    max_length = 0
    for seq in batch:
        if len(seq) > max_length:
            max_length = len(seq)

    new_batch = []
    for seq in batch:
        padding = [pad_idx for i in range(max_length - len(seq))]
        new_batch.append(padding + seq)

    return new_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--length", type=int, default=20, help="Number of tokens to generate for each prompt")
    parser.add_argument("--window_size", type=int, default=None, help="Number of tokens to generate at a time. Note that this only is only implemented for when the text is already written as numerical indices, not when it's written in actual words.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tends toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=40, help="For top-k sampling: Only sample from the k most probable words")
    parser.add_argument("--p", type=float, default=1, help="For top-p sampling (aka nucleus sampling): Only sample from the top p portion of the probability mass")
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size for beam search")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--pretokenized", action="store_true", help="Whether the input text is already tokenized and converted to integer ids")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--prompt_file", type=str, default=None, help="File of prompts, 1 prompt per line.")

    parser.add_argument("--batch_size", type=int, default=10, help="Number of prompts to include in a batch.")
    parser.add_argument("--input_directory", type=str, default=None, help="Directory for input files")
    parser.add_argument("--output_directory", type=str, default=None, help="Directory for output files")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    output_filename = "_".join([str(x) for x in [args.model_name_or_path, args.prompt_file.split("/")[-1], "k" + str(args.k), "p" + str(args.p), "temp" + str(args.temperature), "beam" + str(args.beam_size), "len" + str(args.length), "batchsize" + str(args.batch_size)]]) + ".generated"
    logging.info("Output filename is: " + output_filename)
    fo = open(args.output_directory + output_filename, "w", encoding="utf-8")

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    logger.info(args)

    prompt_file = open(args.input_directory + args.prompt_file, "r", encoding="utf-8")
    prompt_list = prompt_file.readlines()
    prompt_batches = batchify_prompts(prompt_list, args.batch_size)

    for batch_number, prompt_batch in enumerate(prompt_batches):

        logger.info("Processing batch #" + str(batch_number) + " out of " + str(len(prompt_batches)))

        current_prompt = prompt_batch
        generated_sequences = [[] for _ in range(args.batch_size)]
        
        for window_number in range((args.length // args.window_size) + 1):
            if args.pretokenized:
                encoded_prompt = [prmpt.strip().split() for prmpt in current_prompt]
                encoded_prompt = [[int(x) for x in prmpt] for prmpt in encoded_prompt]
                encoded_prompt = pad_batch(encoded_prompt, 50256)
                encoded_prompt = torch.LongTensor(encoded_prompt)
                attention_mask = 1 - (encoded_prompt == 50256).type(torch.LongTensor)
                attention_mask = attention_mask.to(args.device)
            else:
                # Tokenize the prompts and convert to integer ids
                tokenizer.pad_token = "<PADDINGTOKEN>"
                tokenizer.padding_side = "left"

                encoded_prompt = [tokenizer.convert_tokens_to_ids(prompt.strip().split()) for prompt in prompt_batch]
                encoded_prompt = torch.LongTensor(encoded_prompt)

                # Create a dummy attention mask; there is no padding in this case (all prompts
                # are equal lengths), so we don't need to mask
                # 300000 is beyond the vocab size, so ignoring it has no effect
                attention_mask = 1 - (encoded_prompt == 300000).type(torch.LongTensor)
                attention_mask = attention_mask.to(args.device)
        
            encoded_prompt = encoded_prompt.to(args.device)

            if args.model_type == "transfo-xl": 
                # We add 10 to the generation length to give some wiggle
                # room in case of tokenization discrepancies etc.
                output_sequences = model.generate(
                    input_ids=encoded_prompt,
                    max_length=args.window_size + len(encoded_prompt[0] + 10),
                    min_length=args.window_size + len(encoded_prompt[0] + 10),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                    attention_mask=attention_mask,
                    num_beams=args.beam_size,
                    use_cache=True,
                    eos_token_id=267734,
                )

            elif args.model_type == "gpt2":
                # We add 10 to the generation length to give some wiggle room in 
                # case of tokenization discrepancies etc.
                output_sequences = model.generate(
                    input_ids=encoded_prompt,
                    max_length=args.window_size + len(encoded_prompt[0] + 10),
                    min_length=args.window_size + len(encoded_prompt[0] + 10),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                    attention_mask=attention_mask,
                    num_beams=args.beam_size,
                    use_cache=True,
                )
            else:
                print("invalid model type")


            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            if args.pretokenized:
                new_prompt = []

                for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                    new_generation = generated_sequence[len(encoded_prompt[generated_sequence_idx]):].tolist()
                    generated_sequences[generated_sequence_idx] = generated_sequences[generated_sequence_idx] + new_generation
                    new_prompt.append(" ".join([str(x) for x in new_generation]))
                current_prompt = new_prompt[:]

        if args.pretokenized:
            for generated_sequence_idx, generated_sequence in enumerate(generated_sequences):
                fo.write(" ".join([str(x) for x in generated_sequence]) + "\n")
        else:
            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                generated_sequence = generated_sequence.tolist()
                
                # Decode text
                text = tokenizer.convert_ids_to_tokens(output_sequences[generated_sequence_idx], skip_special_tokens=False)

                # Remove the prompt from the beginning of the sequence.
                generated_sequence = (
                    text[len(tokenizer.convert_ids_to_tokens(encoded_prompt[generated_sequence_idx], skip_special_tokens=False)) :]
                )

                generated_sequence = " ".join(generated_sequence)

                fixed_generation = " ".join(ftfy.fix_text(generated_sequence.strip()).split()[:args.length])
                fo.write(fixed_generation + "\n")




if __name__ == "__main__":
    main()
    logger.info("DONE GENERATING")
