import numpy as np
from transformers import GPT2Tokenizer
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_to_decode", type=str, default=None, help="File of BPE tokens to convert to text")
parser.add_argument("--preserve_lines", action="store_true", help="Whether to use the linebreaks from the original file (if omitted, the linebreaks will be the ones gotten from the decoded BPE text)")
args = parser.parse_args()


fi = open(args.file_to_decode, "r")
fo_word = open(args.file_to_decode + ".word", "w")
fo_bpe = open(args.file_to_decode + ".bpe", "w")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def remove_excess_spaces(string):
    starts_with_space = (string.lstrip() != string)
    ends_with_space = (string.rstrip() != string)

    lines = string.split("\n")
    edited_lines = []

    for line in lines:
        edited_lines.append(" ".join(line.split()))
    joined_lines = "\n".join(edited_lines)

    if starts_with_space:
        joined_lines = " " + joined_lines
    if ends_with_space:
        joined_lines = joined_lines + " "

    return joined_lines, starts_with_space, ends_with_space

prev_ends_with_space = False
for line in fi:
    tokens = [int(x) for x in line.strip().split()]
    
    text = tokenizer.decode(tokens)

    # If our newline symbol &NEWLINE; occurs in the text,
    # replace it so that it does not get interpreted
    # as a newline
    # Also convert end-of-document tokens to a newline
    text = text.replace("&NEWLINE;", "NEWLINE").replace(r"<|endoftext|>", "\n")
    
    if args.preserve_lines:
        text, _, _ = remove_excess_spaces(text.replace("\n", " &NEWLINE; "))
        fo_word.write(text + "\n")
    else:
        text_to_write, starts_with_space, ends_with_space = remove_excess_spaces(text)
        if prev_ends_with_space and starts_with_space:
            text_to_write = text_to_write[1:]

        prev_ends_with_space = ends_with_space
        fo_word.write(text_to_write)

    bpes = []
    for token in tokens:
        bp = tokenizer.decode(token).replace("&NEWLINE;", "NEWLINE").replace("\n", "&NEWLINE;").replace(r"<|endoftext|>", "\n")
        if bp[0] == " ":
            bpes.append(bp)
        else:
            bpes.append("@" + bp)
    full_bpes = " ".join(bpes)
    fo_bpe.write(remove_excess_spaces(full_bpes)[0] + "\n")
        


