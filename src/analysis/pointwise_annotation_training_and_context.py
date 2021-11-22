import json
import collections
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--training_pointwise", help="file annotated for pointwise overlap with the training set", type=str, default=None)
parser.add_argument("--context_pointwise", help="file annotated for pointwise overlap with the context", type=str, default=None)
args = parser.parse_args()


training_pointwise = open(args.training_pointwise, "r")
context_pointwise = open(args.context_pointwise, "r")

training_annotated_lines = training_pointwise.readlines()
context_annotated_lines = context_pointwise.readlines()

max_annotated = []

for index, (training_annotated_line, context_annotated_line) in enumerate(zip(training_annotated_lines, context_annotated_lines)):
    ta = training_annotated_line.strip().split()
    pa = context_annotated_line.strip().split()

    max_list = []
    prompt_ended = False

    for ta_pair, pa_pair in zip(ta, pa):
        if ta_pair == "<END_OF_PROMPT>":
            if pa_pair != "<END_OF_PROMPT>":
                print("MISMATCH EOP", index, args.training_pointwise, args.context_pointwise)
            else:
                max_list.append("<END_OF_PROMPT>")
            prompt_ended = True
        else:
            ta_word = "/".join(ta_pair.split("/")[:-1])
            ta_score = int(ta_pair.split("/")[-1])

            pa_word = "/".join(pa_pair.split("/")[:-1])
            pa_score = int(pa_pair.split("/")[-1])

            if ta_word != pa_word:
                print("MISMATCH WORD", index, args.training_pointwise, args.context_pointwise)
            else:
                max_score = max(ta_score, pa_score)

            max_pair = ta_word + "/" + str(max_score)
            max_list.append(max_pair)

    max_annotated.append(max_list)
    

fo_scored = open(args.training_pointwise.replace(".training_pointwise", ".context_and_training_pointwise"), "w")

for max_scored in max_annotated:
    fo_scored.write(" ".join(max_scored) + "\n")




