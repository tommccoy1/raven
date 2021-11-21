from sacremoses import MosesDetokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="file to detokenize", type=str, default=None)
parser.add_argument("--newline", help="newline token", type=str, default=None)
args = parser.parse_args()

fi = open(args.file, "r")
fo = open(args.file + ".detokenized", "w")

md = MosesDetokenizer(lang='en')

for paragraph in fi:
    if args.newline is None:
        lines = [paragraph]

    else:
        lines = paragraph.split(args.newline)

    full_detokenized = []

    for line in lines:
        # Deal with the idiosyncratic words that start with @ signs in Wikitext
        edited = line.replace(" @-@", "@-@").replace("@-@ ", "@-@").replace("@-@", "-")
        edited = edited.replace(" @.@", "@.@").replace("@.@ ", "@.@").replace("@.@", ".")
        edited = edited.replace(" @,@", "@,@").replace("@,@ ", "@,@").replace("@,@", ",")
    
        words = edited.split()

        detokenized = md.detokenize(words)

        full_detokenized.append(detokenized)

    if args.newline:
        fo.write((" " + args.newline + " ").join(full_detokenized).replace("  ", " ") + "\n")
    else:
        fo.write(full_detokenized[0] + "\n")



