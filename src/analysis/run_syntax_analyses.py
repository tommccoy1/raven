
from syntax_analyses import *

import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--training", help="start of all the training filenames (ends with .parsed)", type=str, default=None)
parser.add_argument("--generation", help="start of all the generation filenames (ends with .parsed)", type=str, default=None)
parser.add_argument("--all_analyses", help="run all analyses", action='store_true')
args = parser.parse_args()

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logging.info(args)

# CHANGE BACK
report_file = open(args.generation + ".syntax_report", "w")

logging.info("starting")
all_info_dict = gen_condensed_to_dicts(args.generation)

logging.info("updating")
update_all_info_dict_training(all_info_dict, args.training)

logging.info("counting nonterminal rules")
count_novel_nonterminal_rules, count_nonterminal_rules, novel_nonterminal_rules, novel_unary_binary = analyze_nonterminal_rules(all_info_dict)

logging.info("counting terminal rules")
count_novel_pos_rules, count_pos_rules, novel_pos_rules, novel_nouns, novel_verbs = analyze_terminal_rules(all_info_dict)

logging.info("counting parse structures")
count_novel_parses, count_parses, novel_parses = analyze_parses(all_info_dict)

logging.info("counting POS sequences")
count_novel_pos_seqs, count_pos_seqs, novel_pos_seqs = analyze_pos_seqs(all_info_dict)

logging.info("counting dependency arcs")
count_novel_dep_arcs, count_dep_arcs, novel_dep_arcs, a_to_the, the_to_a, novel_subj, novel_obj = analyze_dep_arcs(all_info_dict)

logging.info("counting dependency paths")
count_novel_dep_paths, count_dep_paths = analyze_dep_paths(all_info_dict)

logging.info("counting unlabeled dependency paths")
count_novel_dep_unlabeled_paths, count_dep_unlabeled_paths = analyze_dep_unlabeled_paths(all_info_dict)

logging.info("counting unlabeled dependency arcs")
count_novel_dep_arcs_unlabeled, count_dep_arcs_unlabeled, novel_dep_arcs_unlabeled = analyze_dep_unlabeled(all_info_dict)

logging.info("counting dependency relations")
count_novel_dep_rels, count_dep_rels, novel_dep_rels, nsubj_to_obj, obj_to_nsubj, active_to_passive, passive_to_active = analyze_dep_rels(all_info_dict)

logging.info("counting dependency argument structures")
count_novel_arg_structures, count_arg_structures, novel_arg_structures, transitive_to_intransitive, intransitive_to_transitive, do_to_po, po_to_do = analyze_dep_arg_structure(all_info_dict)

logging.info("printing results")
if count_pos_rules == 0:
    ratio = 0
else:
    ratio = count_novel_pos_rules*1.0/count_pos_rules

report_file.write("Novel POS tags:\t" + str(ratio) + "\t" + str(count_novel_pos_rules) + "\t" + str(count_pos_rules) + "\n\n")
report_file.write("Examples of novel POS tags:\n")
for example in novel_pos_rules:
    report_file.write(example[0][0] + "\t" + " ".join(example[0][1:]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + ",".join(list(all_info_dict["word2pos_seen"][" ".join(example[0][1:])].keys())) + "\n")
report_file.write("\nExamples of novel nouns:\n")
for example in novel_nouns:
    report_file.write(example[0][0] + "\t" + " ".join(example[0][1:]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + ",".join(list(all_info_dict["word2pos_seen"][" ".join(example[0][1:])].keys())) + "\n")
report_file.write("\nExamples of novel verbs:\n")
for example in novel_verbs:
    report_file.write(example[0][0] + "\t" + " ".join(example[0][1:]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + ",".join(list(all_info_dict["word2pos_seen"][" ".join(example[0][1:])].keys())) + "\n")
report_file.write("\n\n")

if count_nonterminal_rules == 0:
    ratio = 0
else:
    ratio = count_novel_nonterminal_rules*1.0/count_nonterminal_rules

report_file.write("Novel CFG rules:\t" + str(ratio) + "\t" + str(count_novel_nonterminal_rules) + "\t" + str(count_nonterminal_rules) + "\n\n")
report_file.write("Examples of novel CFG rules:\n")
for example in novel_nonterminal_rules:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["const"]) + "\n")
report_file.write("\nExamples of novel unary and binary CFG rules:\n")
for example in novel_unary_binary:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["const"]) + "\n")
report_file.write("\n\n")





if count_pos_seqs == 0:
    ratio = 0
else:
    ratio = count_novel_pos_seqs*1.0/count_pos_seqs

report_file.write("Novel POS sequences:\t" + str(ratio) + "\t" + str(count_novel_pos_seqs) + "\t" + str(count_pos_seqs) + "\n\n")
report_file.write("Examples of novel POS sequences:\n")
for example in novel_pos_seqs:
    report_file.write(example[0] + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["const"]) + "\n")
report_file.write("\n\n")

if count_parses == 0:
    ratio = 0
else:
    ratio = count_novel_parses*1.0/count_parses

report_file.write("Novel constituency structures:\t" + str(ratio) + "\t" + str(count_novel_parses) + "\t" + str(count_parses) + "\n\n")
report_file.write("Examples of novel constituency structures:\n")
for example in novel_parses:
    report_file.write(example[0] + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["const"]) + "\n")
report_file.write("\n\n")

if count_dep_arcs == 0:
    ratio = 0
else:
    ratio = count_novel_dep_arcs*1.0/count_dep_arcs

report_file.write("Novel dependency arcs (labeled):\t" + str(ratio) + "\t" + str(count_novel_dep_arcs) + "\t" + str(count_dep_arcs) + "\n\n")
report_file.write("Examples of novel dependency arcs:\n")
for example in novel_dep_arcs:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\n")
report_file.write("\n\n")
report_file.write("\nExamples of novel dependency arcs - a to the:\n")
for example in a_to_the:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\n")
report_file.write("\nExamples of novel dependency arcs - the to a:\n")
for example in the_to_a:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\n")
report_file.write("\nExamples of novel dependency arcs - novel subject:\n")
for example in novel_subj:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\n")
report_file.write("\nExamples of novel dependency arcs - novel object:\n")
for example in novel_obj:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\n")
report_file.write("\n\n")


if count_dep_arcs_unlabeled == 0:
    ratio = 0
else:
    ratio = count_novel_dep_arcs_unlabeled*1.0/count_dep_arcs_unlabeled

report_file.write("Novel dependency arcs (unlabeled):\t" + str(ratio) + "\t" + str(count_novel_dep_arcs_unlabeled) + "\t" + str(count_dep_arcs_unlabeled) + "\n\n")
report_file.write("Examples of novel dependency arcs (unlabeled):\n") 
for example in novel_dep_arcs_unlabeled:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\n")
report_file.write("\n\n")

report_file.write("Novel dependency paths:\n")
for length in range(1,11):
    if count_dep_paths[length] == 0:
        ratio = 0
    else:
        ratio = count_novel_dep_paths[length]*1.0/count_dep_paths[length]
    report_file.write("Length " + str(length) + ":\t" + str(ratio) + "\t" + str(count_novel_dep_paths[length]) + "\t" + str(count_dep_paths[length]) + "\n")
report_file.write("\n\n")

report_file.write("Novel unlabeled dependency paths:\n")
for length in range(1,11):
    if count_dep_unlabeled_paths[length] == 0:
        ratio = 0
    else:
        ratio = count_novel_dep_unlabeled_paths[length]*1.0/count_dep_unlabeled_paths[length]
    report_file.write("Length " + str(length) + ":\t" + str(ratio) + "\t" + str(count_novel_dep_unlabeled_paths[length]) + "\t" + str(count_dep_unlabeled_paths[length]) + "\n")
report_file.write("\n\n")


if count_dep_rels == 0:
    ratio = 0
else:
    ratio = count_novel_dep_rels*1.0/count_dep_rels

report_file.write("Novel dependency relations:\t" + str(ratio) + "\t" + str(count_novel_dep_rels) + "\t" + str(count_dep_rels) + "\n\n")
report_file.write("Examples of novel dependency relations:\n")
for example in novel_dep_rels:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\n")
report_file.write("\nExamples of novel dependency relations - nsubj to obj:\n")
for example in nsubj_to_obj:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\n")
report_file.write("\nExamples of novel dependency relations - obj to nsubj:\n")
for example in obj_to_nsubj:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join( all_info_dict["indices_to_info"][example[1]]["dep"]) + "\n")
report_file.write("\nExamples of novel dependency relations - active to passive:\n")
for example in active_to_passive:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\n")
report_file.write("\nExamples of novel dependency relations - passive to active:\n")
for example in passive_to_active:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\n")
report_file.write("\n\n")



if count_arg_structures == 0:
    ratio = 0
else:
    ratio = count_novel_arg_structures*1.0/count_arg_structures

report_file.write("Novel dependency argument structures:\t" + str(ratio) + "\t" + str(count_novel_arg_structures) + "\t" + str(count_arg_structures) + "\n\n")
report_file.write("Examples of novel argument structures:\n")
for example in novel_arg_structures:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\t" + " ".join(example[2]) + "\n")
report_file.write("\nExamples of novel argument structures - transitive to intransitive:\n")
for example in transitive_to_intransitive:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\t" + " ".join(example[2]) + "\n")
report_file.write("\nExamples of novel argument structures - intransitive to transitive:\n")
for example in intransitive_to_transitive:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\t" + " ".join(example[2]) + "\n")
report_file.write("\nExamples of novel argument structures - do to po:\n")
for example in do_to_po:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\t" + " ".join(example[2]) + "\n")
report_file.write("\nExamples of novel argument structures - po to do:\n")
for example in po_to_do:
    report_file.write(" ".join(example[0]) + "\t" + all_info_dict["indices_to_info"][example[1]]["sentence"] + "\t" + " ".join(all_info_dict["indices_to_info"][example[1]]["dep"]) + "\t" + " ".join(example[2]) + "\n")
report_file.write("\n\n")





