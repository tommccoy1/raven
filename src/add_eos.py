
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", default=None, type=str, help="File to add EOS tokens to")
args = parser.parse_args()

fi = open(args.file, "r")
fo = open(args.file + ".eos", "w")

for line in fi:
    words = line.strip().split()
    words.append("<eos>")

    fo.write(" ".join(words) + "\n")

