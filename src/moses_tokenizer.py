from sacremoses import MosesTokenizer, MosesDetokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="file to decode", type=str, default=None)
parser.add_argument("--newline", help="token to treat as a newline", type=str, default=None)
args = parser.parse_args()

mt = MosesTokenizer(lang='en')

fi = open(args.file, "r")
fo = open(args.file + ".moses", "w")

for index, line in enumerate(fi):
    if args.newline is None:
        moses_encoded = mt.tokenize(line, return_str=True)
        fo.write(moses_encoded + "\n")
    else:
        sentences = line.split(args.newline)

        encoded_sentence_list = []
        for sentence in sentences:
            moses_encoded = mt.tokenize(sentence, return_str=True)
            encoded_sentence_list.append(moses_encoded)

        sentences_to_print = (" " + args.newline + " ").join(encoded_sentence_list)

        # Handle potential multiple spaces created by the above
        sentences_to_print = " ".join(sentences_to_print.split())

        fo.write(sentences_to_print + "\n")
