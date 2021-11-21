
from analyses import *

import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--pointwise_file", help="file of generations with pointwise copying scores", type=str, default=None)
parser.add_argument("--generation_file", help="file of generated text (without pointwise copying scores", type=str, default=None)
parser.add_argument("--prompt_file", help="file of prompts", type=str, default=None)
parser.add_argument("--training_file", help="file for the training set", type=str, default=None)
parser.add_argument("--perplexity_model", help="model to use for calculating perplexity", type=str, default=None)
parser.add_argument("--eos_token", help="EOS token", type=str, default=None)
parser.add_argument("--all_analyses", help="run all analyses (excludes 'random_counts')", action='store_true')
parser.add_argument("--fast_analyses", help="run only the analyses that are fast (i.e., don't require iterating over the training set or computing perplexity)", action='store_true')
parser.add_argument("--pointwise", help="run pointwise analyses", action='store_true') 
parser.add_argument("--positional", help="run positional analyses", action='store_true')
parser.add_argument("--ngram_overlap", help="run ngram overlap analyses", action='store_true')
parser.add_argument("--novel_bigrams", help="give examples of novel bigrams", action='store_true')
parser.add_argument("--supercopying", help="run supercopying analyses", action='store_true')
parser.add_argument("--supercopying_overlap", help="run analyses of how much supercopying overlaps with the training set", action='store_true') 
parser.add_argument("--random_counts", help="run analyses of how many times random n-grams are repeated in the training set", action='store_true')
parser.add_argument("--perplexity", help="get perplexity", action='store_true')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),logging.FileHandler(args.pointwise_file + "_ngram.log")])
logging.info(args)

if args.all_analyses or args.perplexity:
    if args.perplexity_model == "transfo-xl-wt103":
        gen_perplexity = perplexity_txl(args.generation_file, args.prompt_file)
    else:
        gen_perplexity = perplexity_gpt2(args.generation_file, args.prompt_file)

all_words_and_scores = words_and_scores_from_file(args.pointwise_file)

if args.all_analyses or args.fast_analyses or args.pointwise:
    pointwise_score, inverse_pointwise_score, log2_pointwise_score, ln_pointwise_score, truncated5_pointwise_score = avg_pointwise_novelty(all_words_and_scores)

if args.all_analyses or args.fast_analyses or args.positional:
    average_pointwise_by_position, binned_average_pointwise, truncated_average_pointwise_by_position, truncated_binned_average_pointwise = avg_pointwise_novelty_by_position(all_words_and_scores)

if args.all_analyses or args.fast_analyses or args.ngram_overlap:
    overlap_sizes, overlap_proportions = ngram_overlap_from_pointwise(all_words_and_scores)

if args.all_analyses or args.fast_analyses or args.novel_bigrams:
    novel_bigrams = novel_bigram_examples(all_words_and_scores, 100)

if args.all_analyses or args.fast_analyses or args.supercopying:
    supercopies, longest_supercopy, length_longest_supercopy = supercopying_examples(all_words_and_scores, 5)

if args.all_analyses or args.supercopying_overlap:
    all_supercopying_examples = all_supercopies(all_words_and_scores)
    all_supercopying_examples_full = all_supercopies_full(all_words_and_scores)

    average_supercopy_count, supercopy_count_list, max_supercopy_count, max_supercopy, _ = count_occurrences_in_file(all_supercopying_examples, args.training_file, eos=args.eos_token)
    average_supercopy_max_count, supercopy_max_count_list = count_supercopy_max(all_supercopying_examples, supercopy_count_list, all_supercopying_examples_full)

if args.random_counts:
    random_ngram_examples = random_ngrams(args.training_file, 1000, length=100, eos=args.eos_token)
    average_random_count, random_count_list, max_random_count, max_random, _ = count_occurrences_in_file(random_ngram_examples, args.training_file, eos=args.eos_token)

report_file = open(args.pointwise_file + ".ngram_report", "w")

report_file.write("Generation filename: " + args.generation_file + "\n\n")
if args.all_analyses or args.perplexity:
    report_file.write("Perplexity: " +  str(gen_perplexity) + "\n")

if args.all_analyses or args.fast_analyses or args.pointwise:
    report_file.write("Pointwise score: " + str(pointwise_score) + "\n")
    report_file.write("Inverse pointwise score: " + str(inverse_pointwise_score) + "\n")
    report_file.write("Log2 pointwise score: " +  str(log2_pointwise_score) + "\n")
    report_file.write("Ln pointwise score:" + str(ln_pointwise_score) + "\n")
    report_file.write("Truncated-5 pointwise score:" + str(truncated5_pointwise_score) + "\n\n")

if args.all_analyses or args.fast_analyses or args.positional:
    report_file.write("Average pointwise score by position:" + "\n")
    report_file.write(",".join([str(x) for x in average_pointwise_by_position]) + "\n")
    report_file.write("Binned average pointwise score by position:" + "\n")
    report_file.write(",".join([str(x) for x in binned_average_pointwise]) + "\n")
    report_file.write("Truncated average pointwise score by position:" + "\n")
    report_file.write(",".join([str(x) for x in truncated_average_pointwise_by_position]) + "\n")
    report_file.write("Truncated binned average pointwise score by position:" + "\n")
    report_file.write(",".join([str(x) for x in truncated_binned_average_pointwise]) + "\n\n")

if args.all_analyses or args.fast_analyses or args.ngram_overlap:
    report_file.write("Overlap sizes: " + ",".join([str(x) for x in overlap_sizes]) + "\n")
    report_file.write("Overlap proportions:" + ",".join([str(x) for x in overlap_proportions]) + "\n\n")

if args.all_analyses or args.fast_analyses or args.novel_bigrams:
    report_file.write("Novel bigram examples in context" + "\n")
    for novel_bigram in novel_bigrams:
        report_file.write(novel_bigram + "\n")
    report_file.write("\n")

if args.all_analyses or args.fast_analyses or args.supercopying:
    report_file.write("Supercopying examples in context" + "\n")
    for supercopy in supercopies:
        report_file.write(supercopy + "\n")
    report_file.write("\n")

    report_file.write("Length of longest supercopying example: " + str(length_longest_supercopy) + "\n")
    report_file.write("Longest supercopying example:\n")
    report_file.write(longest_supercopy + "\n\n")

if args.all_analyses or args.supercopying_overlap:
    report_file.write("Average supercopying overlap: " + str(average_supercopy_count) + "\n")
    report_file.write("Average supercopying max overlap: " + str(average_supercopy_max_count) + "\n")

if args.random_counts:
    report_file.write("Average random overlap: " + str(average_random_count) + "\n")

if args.all_analyses or args.supercopying_overlap:
    report_file.write("All supercopying overlaps:\n")
    report_file.write(",".join([str(x) for x in supercopy_count_list]) + "\n")
    report_file.write("All supercopying max overlaps:\n")
    report_file.write(",".join([str(x) for x in supercopy_max_count_list]) + "\n")

if args.random_counts:
    report_file.write("All random overlaps:\n")
    report_file.write(",".join([str(x) for x in random_count_list]) + "\n")

if args.all_analyses or args.supercopying_overlap:
    report_file.write("Max supercopy overlap: " + str(max_supercopy_count) + "\n")
    report_file.write("Max supercopy:" + "\n")
    if max_supercopy is not None:
        report_file.write(" ".join(max_supercopy) + "\n")


