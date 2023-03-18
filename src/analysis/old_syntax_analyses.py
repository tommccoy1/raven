import logging
from random import shuffle

# Input: Constituency parse
# Output: 3 things:
# - List of nonterminal rules that appear in it
# - List of terminal rules that appear in it
# - Sequence of POS tags
def rules_from_parse(parse):
    stack = []
    extra_terminals = []

    nonterminal_rules = []
    terminal_rules = []

    for word in parse:
        if word[0] == "(":
            stack = stack + [word]

        elif word == ")" and extra_terminals == []:
            stack = stack[::-1]
            for index, elt in enumerate(stack):
                if elt[0] == "(":
                    index_rule = index
                    break
            stack = stack[::-1]

            stack[-1*(index_rule + 1)] = stack[-1*(index_rule + 1)][1:]

            rule = tuple(stack[-1*(index_rule + 1):])
            nonterminal_rules.append(rule)
            stack = stack[:-1*(index_rule + 1)+1]

        elif word[-1] == ")":
            stack = stack[::-1]
            for index, elt in enumerate(stack):
                if elt[0] == "(":
                    stack[index] = stack[index][1:]
                    break
            stack = stack[::-1]

            rule = tuple([stack[-1]] + extra_terminals + [word[:-1]])
            terminal_rules.append(rule)
            
            extra_terminals = []

        else:
            extra_terminals.append(word)

    if len(stack) != 1:
        print("Parse", parse)
        print("Stack", stack)
        print("")

    pos_seq = " ".join([rule[0] for rule in terminal_rules])

    return nonterminal_rules, terminal_rules, pos_seq

def ngrams_from_list(words, ngram_size):
    # Gives a list of all ngrams of size ngram_size in line
    ngram_list = zip(*[words[i:] for i in range(ngram_size)])

    return ngram_list

def pos_bigrams_and_trigrams_from_seq(pos_seq):
    poses = pos_seq.split()
    bigram_seq = ["<SOS>"] + poses + ["<EOS>"]

    bigrams = ngrams_from_list(bigram_seq, 2)

    trigram_seq = ["<SOS>"] + bigram_seq
    trigrams = ngrams_from_list(trigram_seq, 3)

    trigrams_b = []
    rule = []
    for index, elt in enumerate(trigram_seq):
        rule.append(elt)

        if len(rule) == 3:
            trigrams_b.append(tuple(rule))
            rule = [rule[-1]]

    if len(rule) > 0:
        if len(rule) == 1:
            rule = tuple(rule + ["<EOS>", "<EOS>"])
        elif len(rule) == 2:
            rule = tuple(rule + ["<EOS>"])
        else:
            14/0

        trigrams_b.append(rule)


    fourgrams_seq = ["<SOS>"] + trigram_seq
    fourgrams = ngrams_from_list(fourgrams_seq, 4)

    return bigrams, trigrams, fourgrams, trigrams_b

def grammar_counts_from_file(filename, gen=True):
    fi = open(filename, "r")

    cfg_dict = {}
    pos_bigram_dict = {}
    pos_trigram_dict = {}
    pos_fourgram_dict = {}

    pos_trigram_tree_dict = {}

    for index, line in enumerate(fi):
        if index % 100000 == 0:
            logging.info(index)

        parts = line.strip().split("\t")
        sentence = parts[0]
        const = parts[1]
        dep = parts[2]

        if gen:
            line_index = int(parts[-3])
            first_gen_index = int(parts[-2])
            sent_index = int(parts[-1])

            if sent_index < first_gen_index:
                continue

        for const_parse in const.split("&"):
            const_parse = "(ROOT " + const_parse + " )"
            const_parse = const_parse.split()
            nonterminal_rules, terminal_rules, pos_seq = rules_from_parse(const_parse)
            pos_bigrams, pos_trigrams, pos_fourgrams, pos_trigrams_tree = pos_bigrams_and_trigrams_from_seq(pos_seq)

            for rule in nonterminal_rules:
                if rule not in cfg_dict:
                    cfg_dict[rule] = 0
                cfg_dict[rule] += 1

            for rule in terminal_rules:
                rule = (rule[0], "w")
                if rule not in cfg_dict:
                    cfg_dict[rule] = 0
                cfg_dict[rule] += 1

            for pos_bigram in pos_bigrams:
                if pos_bigram not in pos_bigram_dict:
                    pos_bigram_dict[pos_bigram] = 0
                pos_bigram_dict[pos_bigram] += 1

            for pos_trigram in pos_trigrams:
                if pos_trigram not in pos_trigram_dict:
                    pos_trigram_dict[pos_trigram] = 0
                pos_trigram_dict[pos_trigram] += 1

            for pos_fourgram in pos_fourgrams:
                if pos_fourgram not in pos_fourgram_dict:
                    pos_fourgram_dict[pos_fourgram] = 0
                pos_fourgram_dict[pos_fourgram] += 1

            for pos_trigram_tree in pos_trigrams_tree:
                if pos_trigram_tree not in pos_trigram_tree_dict:
                    pos_trigram_tree_dict[pos_trigram_tree] = 0
                pos_trigram_tree_dict[pos_trigram_tree] += 1


    return cfg_dict, pos_bigram_dict, pos_trigram_dict, pos_fourgram_dict, pos_trigram_tree_dict

def abstract_parse_from_parse(parse):
    new_parse = []

    for word in parse:
        if word[-1] == ")" and len(word) > 1:
            new_parse.append("w)")
        elif word[0] != "(" and word[-1] != ")":
            new_parse.append("w")
        else:
            new_parse.append(word)

    new_parse = " ".join(new_parse)

    return new_parse

def dep_arcs_from_dep(arc_list):
    triples = []
    rels = []
    unlabeled_arcs = []

    for arc in arc_list:
        parts = arc.split(",")
        rel = parts[0]
        word1 = parts[1]
        word2 = parts[3]

        triples.append((rel, word1, word2))
        unlabeled_arcs.append((word1, word2))

        rel1 = (word1, "_" + rel)
        rel2 = (word2, rel + "_")

        rels.append(rel1)
        rels.append(rel2)

    return triples, rels, unlabeled_arcs

def dep_arg_structure_from_dep(arc_list):

    verbs = {}
    pobjs = {}
    preps = {}

    structure_list = []

    for arc in arc_list:
        parts = arc.split(",")
        rel = parts[0]
        w1 = parts[1]
        ind1 = parts[2]
        w2 = parts[3]
        ind2 = parts[4]

        if (rel == "dobj" or rel == "iobj" or rel == "nsubjpass" or rel == "nsubj"): 
            verb = (w2, ind2)
                
            if verb not in verbs:
                verbs[verb] = []
            verbs[verb].append(rel)

        # Just need to register that the preposition has an object
        elif rel == "pobj":
            prep = (w2, ind2)
            pobjs[prep] = 1

        elif rel == "prep":
            prep = (w1, ind1)
            verb = (w2, ind2)
            if verb not in preps:
                preps[prep] = verb

    # Add prepositional arguments (or adjuncts) to verbs
    for prep in preps:

        # Only consider the preposition if it has an object
        if prep in pobjs:
            verb = preps[prep]
            
            # Only consider a verb if it has some other argument
            # already; otherwise the "verb" is probably something
            # else that is modified by a PP (e.g., a noun)
            if verb in verbs:
                verbs[verb].append("pobj")

    for verb in verbs:
        args = ",".join(sorted(verbs[verb]))
        args = "_".join(args.split())

        structure_list.append((verb[0], args))

    return structure_list

def gen_condensed_to_dicts(gen_condensed_filename):
    nonterminal_rules_dict = {}
    terminal_rules_dict = {}
    word2pos_seen = {}
    word_seen_in_dep = {}
    word2det_seen = {}
    parse_dict = {}
    pos_seq_dict = {}
    dep_arc_dict = {}
    dep_rel_dict = {}
    dep_unlabeled_dict = {}
    dep_arg_structure_dict = {}

    indices_to_info = {}

    fi = open(gen_condensed_filename, "r")

    for index, line in enumerate(fi):
        if index % 1000 == 0:
            logging.info(index)

        parts = line.strip().split("\t")
        sentence = parts[0]
        const = parts[1]
        dep = parts[2]
        line_index = int(parts[3])
        first_gen_index = int(parts[4])
        sentence_index = int(parts[5])

        position = (line_index, first_gen_index, sentence_index)
        indices_to_info[position] = {}
        indices_to_info[position]["sentence"] = sentence
        indices_to_info[position]["const"] = const.split()
        indices_to_info[position]["dep"] = dep.split()

        for const_parse in const.split("&"):
            const_parse = const_parse.split()
            nonterminal_rules, terminal_rules, pos_seq = rules_from_parse(const_parse)
            abstract_parse = abstract_parse_from_parse(const_parse)

            for nonterminal_rule in nonterminal_rules:
                if nonterminal_rule not in nonterminal_rules_dict:
                    nonterminal_rules_dict[nonterminal_rule] = [[], False]
                nonterminal_rules_dict[nonterminal_rule][0].append(position)

            for terminal_rule in terminal_rules:
                if terminal_rule not in terminal_rules_dict:
                    terminal_rules_dict[terminal_rule] = [[], False]
                terminal_rules_dict[terminal_rule][0].append(position)

                tag = terminal_rule[0]
                word = " ".join(terminal_rule[1:])
                word2pos_seen[word] = {}

            if abstract_parse not in parse_dict:
                parse_dict[abstract_parse] = [[], False]
            parse_dict[abstract_parse][0].append(position)

            if pos_seq not in pos_seq_dict:
                pos_seq_dict[pos_seq] = [[], False]
            pos_seq_dict[pos_seq][0].append(position)

        for dep_parse in dep.split("&"):
            dep_parse = dep_parse.split()
            dep_triples, dep_rels, dep_unlabeleds = dep_arcs_from_dep(dep_parse)
            dep_arg_structures = dep_arg_structure_from_dep(dep_parse)

       
            for dep_triple in dep_triples:
                if dep_triple not in dep_arc_dict:
                    dep_arc_dict[dep_triple] = [[], False]
                dep_arc_dict[dep_triple][0].append(position)
            
                if dep_triple[0] == "det":
                    word2det_seen[dep_triple[2]] = {}

                _, w1, w2 = dep_triple
                word_seen_in_dep[w1] = False
                word_seen_in_dep[w2] = False

            for (word, rel) in dep_rels:
                if word not in dep_rel_dict:
                    dep_rel_dict[word] = {}
                    dep_rel_dict[word]["TRAINING"] = {}
                    dep_rel_dict[word]["GENERATION"] = {}

                if rel not in dep_rel_dict[word]["GENERATION"]:
                    dep_rel_dict[word]["GENERATION"][rel] = [[], False]

                dep_rel_dict[word]["GENERATION"][rel][0].append(position)
            
            for dep_unlabeled in dep_unlabeleds:
                if dep_unlabeled not in dep_unlabeled_dict:
                    dep_unlabeled_dict[dep_unlabeled] = [[], False]
                dep_unlabeled_dict[dep_unlabeled][0].append(position)


            for (verb, arg_structure) in dep_arg_structures:
                if verb not in dep_arg_structure_dict:
                    dep_arg_structure_dict[verb] = {}
                    dep_arg_structure_dict[verb]["TRAINING"] = {}
                    dep_arg_structure_dict[verb]["GENERATION"] = {}

                if arg_structure not in dep_arg_structure_dict[verb]["GENERATION"]:
                    dep_arg_structure_dict[verb]["GENERATION"][arg_structure] = [[], False]

                dep_arg_structure_dict[verb]["GENERATION"][arg_structure][0].append(position)

    all_info_dict = {}
    all_info_dict["nonterminal_rules"] = nonterminal_rules_dict
    all_info_dict["terminal_rules"] = terminal_rules_dict
    all_info_dict["word2pos_seen"] = word2pos_seen
    all_info_dict["word_seen_in_dep"] = word_seen_in_dep
    all_info_dict["word2det_seen"] = word2det_seen 
    all_info_dict["parses"] = parse_dict
    all_info_dict["pos_seqs"] = pos_seq_dict
    all_info_dict["dep_arcs"] = dep_arc_dict
    all_info_dict["dep_rels"] = dep_rel_dict
    all_info_dict["dep_unlabeled"] = dep_unlabeled_dict
    all_info_dict["dep_arg_structures"] = dep_arg_structure_dict
    all_info_dict["indices_to_info"] = indices_to_info

    return all_info_dict

def update_all_info_dict_training(all_info_dict, training_filename):
    fi = open(training_filename, "r")

    for index, line in enumerate(fi):
        if index % 100000 == 0:
            logging.info(index)

        parts = line.strip().split("\t")
        sentence = parts[0]
        const = parts[1]
        dep = parts[2]

        for const_parse in const.split("&"):
            const_parse = const_parse.split()
            nonterminal_rules, terminal_rules, pos_seq = rules_from_parse(const_parse)
            abstract_parse = abstract_parse_from_parse(const_parse)
    
            for nonterminal_rule in nonterminal_rules:
                if nonterminal_rule in all_info_dict["nonterminal_rules"]:
                    all_info_dict["nonterminal_rules"][nonterminal_rule][1] = True

            for terminal_rule in terminal_rules:
                if terminal_rule in all_info_dict["terminal_rules"]:
                    all_info_dict["terminal_rules"][terminal_rule][1] = True

                tag = terminal_rule[0]
                word = " ".join(terminal_rule[1:])
                if word in all_info_dict["word2pos_seen"]:
                    all_info_dict["word2pos_seen"][word][tag] = 1

            if abstract_parse in all_info_dict["parses"]:
                all_info_dict["parses"][abstract_parse][1] = True

            if pos_seq in all_info_dict["pos_seqs"]:
                all_info_dict["pos_seqs"][pos_seq][1] = True

        for dep_parse in dep.split("&"):
            dep_parse = dep_parse.split()
            dep_triples, dep_rels, dep_unlabeleds = dep_arcs_from_dep(dep_parse)
            dep_arg_structures = dep_arg_structure_from_dep(dep_parse)

            for dep_triple in dep_triples:
                if dep_triple in all_info_dict["dep_arcs"]:
                    all_info_dict["dep_arcs"][dep_triple][1] = True

                if dep_triple[1] in all_info_dict["word_seen_in_dep"]:
                    all_info_dict["word_seen_in_dep"][dep_triple[1]] = True
                if dep_triple[2] in all_info_dict["word_seen_in_dep"]:
                    all_info_dict["word_seen_in_dep"][dep_triple[2]] = True

                # Keeping track of which determiners each word has appeared with
                if dep_triple[0] == "det" and dep_triple[2] in all_info_dict["word2det_seen"]:
                    all_info_dict["word2det_seen"][dep_triple[2]][dep_triple[1].lower()] = 1

            for (word, rel) in dep_rels:
                if word in all_info_dict["dep_rels"]:
                    # We only store examples for these 4 types of relations
                    if rel == "nsubj_" or rel == "nsubjpass_" or rel == "_dobj" or rel == "_nsubj":
                        all_info_dict["dep_rels"][word]["TRAINING"][rel] = 1

                    if rel in all_info_dict["dep_rels"][word]["GENERATION"]:
                        all_info_dict["dep_rels"][word]["GENERATION"][rel][1] = True

            for dep_unlabeled in dep_unlabeleds:
                if dep_unlabeled in all_info_dict["dep_unlabeled"]:
                    all_info_dict["dep_unlabeled"][dep_unlabeled][1] = True


            for (verb, arg_structure) in dep_arg_structures:
                if verb in all_info_dict["dep_arg_structures"]:
                    # PLACE THREE
                    all_info_dict["dep_arg_structures"][verb]["TRAINING"][arg_structure] = 1

                    if arg_structure in all_info_dict["dep_arg_structures"][verb]["GENERATION"]:
                        all_info_dict["dep_arg_structures"][verb]["GENERATION"][arg_structure][1] = True



def count_generation_appearances(appearance_list):
    gen_appearances = [appearance for appearance in appearance_list if appearance[1] <= appearance[2]]

    return len(gen_appearances)

def count_first_appearances_in_generations(appearance_list):
    line2sentences = {}

    for appearance in appearance_list:
        line_index, first_gen_index, sentence_index = appearance
        if line_index not in line2sentences:
            line2sentences[line_index] = []
        line2sentences[line_index].append((int(first_gen_index), int(sentence_index)))

    count_first_from_gens = 0
    first_appearance_list = []
    for line_index in line2sentences:
        sorted_appearances = sorted(line2sentences[line_index], key=lambda x: x[1])
        if sorted_appearances[0][1] >= sorted_appearances[0][0]:
            count_first_from_gens += 1
            first_appearance_list.append((line_index, sorted_appearances[0][0], sorted_appearances[0][1]))

    return count_first_from_gens, first_appearance_list

def analyze_nonterminal_rules(all_info_dict):
    novel_rules = []
    novel_unary_binary = []

    count_novel_rules = 0
    count_rules = 0

    for nonterminal_rule in all_info_dict["nonterminal_rules"]:
        count_rules += count_generation_appearances(all_info_dict["nonterminal_rules"][nonterminal_rule][0])
        if not all_info_dict["nonterminal_rules"][nonterminal_rule][1]:
            count_first_from_gens, first_appearance_list = count_first_appearances_in_generations(all_info_dict["nonterminal_rules"][nonterminal_rule][0])
            count_novel_rules += count_first_from_gens

            for first_appearance in first_appearance_list:
                novel_rules.append((nonterminal_rule, first_appearance))

                if len(nonterminal_rule) <= 3:
                    novel_unary_binary.append((nonterminal_rule, first_appearance))

    shuffle(novel_rules)

    return count_novel_rules, count_rules, novel_rules[:100], novel_unary_binary


def analyze_terminal_rules(all_info_dict):
    novel_rules = []
    novel_nouns = []
    novel_verbs = []

    count_novel_rules = 0
    count_rules = 0

    for terminal_rule in all_info_dict["terminal_rules"]:
        # We only consider words that appeared in training;
        # if they haven't appeared in training, then it's 
        # trivial that they'll have a novel POS tag
        if len(all_info_dict["word2pos_seen"][" ".join(terminal_rule[1:])]) == 0:
            continue

        count_rules += count_generation_appearances(all_info_dict["terminal_rules"][terminal_rule][0])

        if not all_info_dict["terminal_rules"][terminal_rule][1]:
            count_first_from_gens, first_appearance_list = count_first_appearances_in_generations(all_info_dict["terminal_rules"][terminal_rule][0])
            count_novel_rules += count_first_from_gens

            for first_appearance in first_appearance_list:
                novel_rules.append((terminal_rule, first_appearance))

                pos = terminal_rule[0]
                word = " ".join(terminal_rule[1:])

                if pos[0] == "N":
                    appears_in_training = False
                    has_n = False
                
                    for training_pos in all_info_dict["word2pos_seen"][word]:
                        appears_in_training = True
                        if training_pos[0] == "N":
                            has_n = True

                    if appears_in_training and not has_n:
                        novel_nouns.append((terminal_rule, first_appearance))
            
                if pos[0] == "V":
                    appears_in_training = False
                    has_v = False
                
                    for training_pos in all_info_dict["word2pos_seen"][word]:
                        appears_in_training = True
                        if training_pos[0] == "V":
                            has_v = True

                    if appears_in_training and not has_v:
                        novel_verbs.append((terminal_rule, first_appearance))

    shuffle(novel_rules)

    return count_novel_rules, count_rules, novel_rules[:200], novel_nouns, novel_verbs

def analyze_parses(all_info_dict):
    novel_parses = []

    count_novel_parses = 0
    count_parses = 0

    for parse in all_info_dict["parses"]:
        count_parses += count_generation_appearances(all_info_dict["parses"][parse][0])

        if not all_info_dict["parses"][parse][1]:
            count_first_from_gens, first_appearance_list = count_first_appearances_in_generations(all_info_dict["parses"][parse][0])
            count_novel_parses += count_first_from_gens

            for first_appearance in first_appearance_list:
                novel_parses.append((parse, first_appearance))

    shuffle(novel_parses)

    return count_novel_parses, count_parses, novel_parses[:100]


def analyze_pos_seqs(all_info_dict):
    novel_pos_seqs = []

    count_novel_pos_seqs = 0
    count_pos_seqs = 0

    for pos_seq in all_info_dict["pos_seqs"]:
        count_pos_seqs += count_generation_appearances(all_info_dict["pos_seqs"][pos_seq][0])

        if not all_info_dict["pos_seqs"][pos_seq][1]:
            count_first_from_gens, first_appearance_list = count_first_appearances_in_generations(all_info_dict["pos_seqs"][pos_seq][0])
            count_novel_pos_seqs += count_first_from_gens

            for first_appearance in first_appearance_list:
                novel_pos_seqs.append((pos_seq, first_appearance))

    shuffle(novel_pos_seqs)

    return count_novel_pos_seqs, count_pos_seqs, novel_pos_seqs[:100]


def analyze_dep_arcs(all_info_dict):
    novel_dep_arcs = []
    a_to_the = []
    the_to_a = []
    novel_subj = []
    novel_obj = []

    count_novel_dep_arcs = 0
    count_dep_arcs = 0

    for dep_arc in all_info_dict["dep_arcs"]:
        count_dep_arcs += count_generation_appearances(all_info_dict["dep_arcs"][dep_arc][0])

        if not all_info_dict["dep_arcs"][dep_arc][1]:
            count_first_from_gens, first_appearance_list = count_first_appearances_in_generations(all_info_dict["dep_arcs"][dep_arc][0])

            count_novel_dep_arcs += count_first_from_gens

            for first_appearance in first_appearance_list:
                novel_dep_arcs.append((dep_arc, first_appearance))

                if dep_arc[0] == "det":
                    if dep_arc[1].lower() not in all_info_dict["word2det_seen"][dep_arc[2]]:
                        if dep_arc[1].lower() == "the" and ("a" in all_info_dict["word2det_seen"][dep_arc[2]] or "an" in all_info_dict["word2det_seen"][dep_arc[2]]):
                            a_to_the.append((dep_arc, first_appearance))

                        if dep_arc[1].lower() == "a" and "an" not in all_info_dict["word2det_seen"][dep_arc[2]] and "the" in all_info_dict["word2det_seen"][dep_arc[2]]:
                            the_to_a.append((dep_arc, first_appearance))

                        if dep_arc[1].lower() == "an" and "a" not in all_info_dict["word2det_seen"][dep_arc[2]] and "the" in all_info_dict["word2det_seen"][dep_arc[2]]:
                            the_to_a.append((dep_arc, first_appearance))

                if dep_arc[0] == "nsubj" and all_info_dict["word_seen_in_dep"][dep_arc[2]]:
                    novel_subj.append((dep_arc, first_appearance))

                if dep_arc[0] == "dobj" and all_info_dict["word_seen_in_dep"][dep_arc[2]]:
                    novel_obj.append((dep_arc, first_appearance))
                        
    shuffle(novel_dep_arcs)
    shuffle(novel_subj)
    shuffle(novel_obj)

    return count_novel_dep_arcs, count_dep_arcs, novel_dep_arcs[:200], a_to_the, the_to_a, novel_subj, novel_obj


def analyze_dep_rels(all_info_dict):
    novel_dep_rels = []
    nsubj_to_dobj = []
    dobj_to_nsubj = []
    active_to_passive = []
    passive_to_active = []

    count_novel_dep_rels = 0
    count_dep_rels = 0

    for word in all_info_dict["dep_rels"]:
        # Only consider if the word appeared in the training set
        if not all_info_dict["word_seen_in_dep"][word]:
            continue
        
        for rel in all_info_dict["dep_rels"][word]["GENERATION"]:
            count_dep_rels += count_generation_appearances(all_info_dict["dep_rels"][word]["GENERATION"][rel][0])

            if not all_info_dict["dep_rels"][word]["GENERATION"][rel][1]:
                
                count_first_from_gens, first_appearance_list = count_first_appearances_in_generations(all_info_dict["dep_rels"][word]["GENERATION"][rel][0])

                count_novel_dep_rels += count_first_from_gens

                for first_appearance in first_appearance_list:
                    novel_dep_rels.append(((word, rel), first_appearance))

                    if rel == "_nsubj":
                        if "_dobj" in all_info_dict["dep_rels"][word]["TRAINING"]:
                            dobj_to_nsubj.append(((word, rel), first_appearance))

                    if rel == "_dobj":
                        if "_nsubj" in all_info_dict["dep_rels"][word]["TRAINING"]:
                            nsubj_to_dobj.append(((word, rel), first_appearance))
 
                    if rel == "nsubj_":
                        if "nsubjpass_" in all_info_dict["dep_rels"][word]["TRAINING"]:
                            passive_to_active.append(((word, rel), first_appearance))

                    if rel == "nsubjpass_":
                        if "nsubj_" in all_info_dict["dep_rels"][word]["TRAINING"]:
                            active_to_passive.append(((word, rel), first_appearance))

    shuffle(novel_dep_rels)
    
    return count_novel_dep_rels, count_dep_rels, novel_dep_rels, nsubj_to_dobj, dobj_to_nsubj, active_to_passive, passive_to_active

def analyze_dep_unlabeled(all_info_dict):
    novel_dep_arcs = []

    count_novel_dep_arcs = 0
    count_dep_arcs = 0

    for dep_arc in all_info_dict["dep_unlabeled"]:
        count_dep_arcs += count_generation_appearances(all_info_dict["dep_unlabeled"][dep_arc][0])

        if not all_info_dict["dep_unlabeled"][dep_arc][1]:
            count_first_from_gens, first_appearance_list = count_first_appearances_in_generations(all_info_dict["dep_unlabeled"][dep_arc][0])

            count_novel_dep_arcs += count_first_from_gens

            for first_appearance in first_appearance_list:
                novel_dep_arcs.append((dep_arc, first_appearance))

    shuffle(novel_dep_arcs)

    return count_novel_dep_arcs, count_dep_arcs, novel_dep_arcs[:200]


def analyze_dep_arg_structure(all_info_dict):
    novel_arg_structures = []
    transitive_to_intransitive = []
    intransitive_to_transitive = []
    do_to_po = []
    po_to_do = []

    count_novel_arg_structures = 0
    count_arg_structures = 0

    for verb in all_info_dict["dep_arg_structures"]:
        # Only consider if the verb appeared in the training set
        if not all_info_dict["word_seen_in_dep"][verb]:
            continue

        for arg_structure in all_info_dict["dep_arg_structures"][verb]["GENERATION"]:

            count_arg_structures += count_generation_appearances(all_info_dict["dep_arg_structures"][verb]["GENERATION"][arg_structure][0])

            if not all_info_dict["dep_arg_structures"][verb]["GENERATION"][arg_structure][1]:
                count_first_from_gens, first_appearance_list = count_first_appearances_in_generations(all_info_dict["dep_arg_structures"][verb]["GENERATION"][arg_structure][0])

                count_novel_arg_structures += count_first_from_gens

                for first_appearance in first_appearance_list:
                    novel_arg_structures.append(((verb, arg_structure), first_appearance, all_info_dict["dep_arg_structures"][verb]["TRAINING"]))

                    if arg_structure == "nsubj" and "dobj,nsubj" in all_info_dict["dep_arg_structures"][verb]["TRAINING"]:
                        transitive_to_intransitive.append(((verb, arg_structure), first_appearance, all_info_dict["dep_arg_structures"][verb]["TRAINING"]))

                    if arg_structure == "dobj,nsubj" and "nsubj" in all_info_dict["dep_arg_structures"][verb]["TRAINING"]:
                        intransitive_to_transitive.append(((verb, arg_structure), first_appearance, all_info_dict["dep_arg_structures"][verb]["TRAINING"]))

                    if arg_structure == "dobj,nsubj,pobj" and "dobj,iobj,nsubj" in all_info_dict["dep_arg_structures"][verb]["TRAINING"]:
                        do_to_po.append(((verb, arg_structure), first_appearance, all_info_dict["dep_arg_structures"][verb]["TRAINING"]))

                    if arg_structure == "dobj,iobj,nsubj" and "dobj,nsubj,pobj" in all_info_dict["dep_arg_structures"][verb]["TRAINING"]:
                        po_to_do.append(((verb, arg_structure), first_appearance, all_info_dict["dep_arg_structures"][verb]["TRAINING"]))

    shuffle(novel_arg_structures)

    return count_novel_arg_structures, count_arg_structures, novel_arg_structures[:200], transitive_to_intransitive, intransitive_to_transitive, do_to_po, po_to_do



if False:
    fi = open("../../data/prompts_and_generations/gpt2-xl_webtext_prompts_length141_1of1_k40_p1_temp1.0_beam1_len1300_batchsize5.generated.trimmed.word.sentences.parsed", "r")

if False:
    for line in fi:
        parts = line.split("\t")
        parses = parts[1].split("&")
        #for parse in parses:
        #    print(parse)
            #nonterminal_rules, terminal_rules, pos_seq = rules_from_parse(parse.split())
            #for rule in rules:
                #print(rule)
            #    pass
            #print(pos_seq)

            #print("")

        #    abstract = abstract_parse_from_parse(parse.split())
        #    print(abstract)
        #    print("")

        deps = parts[2].split("&")
        for dep in deps:
            dep_arcs = dep_arcs_from_dep(dep.split())

            #dep_arg_structures = dep_arg_structure_from_dep(dep.split())
            print(parts[0])
            print(dep)
            #print(dep_arg_structures)
            print(dep_arcs)
            print("")


