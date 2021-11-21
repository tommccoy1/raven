import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="file to trim", type=str, default=None)
parser.add_argument("--length", help="number of tokens to trim to", type=int, default=None)
args = parser.parse_args()

fi = open(args.file, "r")
fo = open(args.file + ".trimmed", "w")

for line in fi:
    words = line.strip().split()
    if len(words) < args.length:
        print(args.file, "Too short:", len(words))
    fo.write(" ".join(words[:args.length]) + "\n")


