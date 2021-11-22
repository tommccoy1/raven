
import argparse
import logging
import benepar, spacy
import nltk
from supar import Parser


parser = argparse.ArgumentParser()
parser.add_argument("--file_to_parse", type=str, default=None, help="File of sentences, one sentence per line")
parser.add_argument("--training", action="store_true", help="Whether it's a training file (in which case line numbers will be ignored)")
args = parser.parse_args()


# Deal with encoding issues
def fix_string(sentence):
    new_sentence = sentence[:].encode("ascii", "ignore").decode()
    new_sentence = remove_spaces(new_sentence.strip())

    return new_sentence


# Remove duplicate spaces
def remove_spaces(string):
    if "  " in string:
        return remove_spaces(string.replace("  ", " "))
    else:
        return string

# Make sure there are spaces around the parentheses
def format_parse(parse):
    parse = parse.replace("(", " (")
    parse = parse.replace(")", ") ")
    parse = remove_spaces(parse)
    parse = parse.strip()

    return parse

def delimiter_format(string):
    space_replaced = "_".join(string.split())
    return space_replaced.replace(",", "-COMMA-").replace("&", "-AMPERSAND-")


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

const_parser = spacy.load('en_core_web_md')
benepar.download('benepar_en3', download_dir='.venv/nltk_data')

if spacy.__version__.startswith('2'):
    const_parser.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    const_parser.add_pipe("benepar", config={"model": "benepar_en3"})


dep_parser = Parser.load('biaffine-dep-en')

fi = open(args.file_to_parse, "r")
fo = open(args.file_to_parse + ".parsed", "w")

for index, line in enumerate(fi):
    if args.training:
        line = fix_string(line)
    else:
        parts = line.strip().split("\t")
        generation_number = parts[-3]
        first_generation_index = parts[-2]
        sentence_index = parts[-1]

        line = fix_string("\t".join(parts[:-3]))

    if line.strip() == "":
        continue


    if index % 1000 == 0:
        logging.info("Done with sentence #" + str(index))
    const_parse = const_parser(line.strip())
    const_sents = list(const_parse.sents)
    all_const_parses = []

    for sent in const_sents:
        all_const_parses.append(format_parse(sent._.parse_string).replace("&", "-AMPERSAND-"))
    const_connected = "&".join(all_const_parses)

    dep_parse = dep_parser.predict([line.strip()], lang='en', verbose=False)

    word_lists = dep_parse.words
    rel_lists = dep_parse.rels
    arc_lists = dep_parse.arcs

    all_dep_parses = []

    for word_list, rel_list, arc_list in zip(word_lists, rel_lists, arc_lists):
        words = ["ROOT"] + list(word_list)
        dep_parse = []
        for index, (word, rel, arc) in enumerate(zip(word_list, rel_list, arc_list)):
            word1 = word 
            index1 = (index+1)
            word2 = words[arc]
            index2 = arc
            full_arc = ",".join([rel, delimiter_format(word1), str(index1), delimiter_format(word2), str(index2)])

            dep_parse.append(full_arc)

        all_dep_parses.append(" ".join(dep_parse))

    dep_connected = "&".join(all_dep_parses)
    
    if args.training:
        fo.write("\t".join([line.strip(), const_connected, dep_connected]) + "\n")
    else:
        fo.write("\t".join([line.strip(), const_connected, dep_connected, generation_number, first_generation_index, sentence_index]) + "\n")
