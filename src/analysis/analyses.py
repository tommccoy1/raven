
import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TransfoXLLMHeadModel, TransfoXLTokenizer

import json
import collections
import math
import random
from random import shuffle
import logging

from sacremoses import MosesDetokenizer 

md = MosesDetokenizer(lang='en')

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Some heuristics for roughly un-doing the tokenization
# that was applied to Wikitext-103
def wikitext_detokenize(passage):
    passage = passage.replace(" <eos> ", "\n").replace("<eos> ", "\n").replace("<eos>", "\n").replace("<unk>", "unk")
    edited = passage.replace(" @-@", "@-@").replace("@-@ ", "@-@").replace("@-@", "-")
    edited = edited.replace(" @.@", "@.@").replace("@.@ ", "@.@").replace("@.@", ".")
    edited = edited.replace(" @,@", "@,@").replace("@,@ ", "@,@").replace("@,@", ",")
   
    words = edited.split()

    detokenized = md.detokenize(words)

    return detokenized

# Replace the token NEWLINE with an actual newline
def replace_newlines(text):
    text_new = text.replace(" &NEWLINE; ", "\n").replace(" &NEWLINE;", "\n").replace("&NEWLINE; ", "\n").replace("\n", " <eos> ")
    return text_new

# Get perplexity using Transformer-XL
def perplexity_txl(file_to_score, prompt_file):

    fi = open(file_to_score, "r")

    total_perplexity = 0
    count_perplexities = 0

    prompts = [x.strip() for x in open(prompt_file, "r").readlines()]

    tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
    model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103").to(device)

    # Minimum number of context tokens that each token must have
    # (as long as there are enough previous tokens to make this
    # possible)
    stride = 512

    # Maximum length sequence of tokens to be processed
    # at once by the model
    max_length = 1024

    for index, line in enumerate(fi):
        # Encode the prompt
        prompt = replace_newlines(prompts[index])
        prompt_tokens = tokenizer.encode(prompt)
        prompt_length = len(prompt_tokens)

        # Encode the generated text
        tokens_line = tokenizer.encode(replace_newlines(line))
        generation_length = len(tokens_line)

        all_tokens = prompt_tokens + tokens_line 
        targets = all_tokens[:]

        # Ignore the prompt tokens in the perplexity calculation
        for i in range(prompt_length):
            targets[i] = -100

        this_ll = 0
        for i in range(0, len(all_tokens), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, len(all_tokens))
            trg_len = end_loc - i    # may be different from stride on last loop
            input_ids = torch.LongTensor(all_tokens[begin_loc:end_loc]).to(device)
            target_ids = torch.LongTensor(targets[begin_loc:end_loc]).to(device)

            # Mask out the ones that have already been scored
            target_ids[:-trg_len] = -100

            with torch.no_grad():
                # Outputs are a list of log likelihoods for all tokens except the first one
                # They will be left-aligned: the first one will be the first non-masked
                # perplexity (i.e., the first non-prompt token). So the zeroes
                # corresponding to the prompt token will be at the end, not the beginning,
                # even though the prompt is at the beginning
                outputs = model(input_ids.unsqueeze(0), labels=target_ids.unsqueeze(0))
                log_likelihood = torch.sum(outputs[0])

            this_ll += log_likelihood
            
        this_ll = this_ll / generation_length

        # Above we averaged the negative log likelihood; so
        # here we just have to exponentiate it to get the perplexity
        ppl = torch.exp(this_ll)
        total_perplexity += ppl.item()
        count_perplexities += 1
        logging.info("RUNNING AVERAGE PERPLEXITY AFTER LINE " + str(index) + ": " + str(total_perplexity * 1.0 / count_perplexities))

    average_perplexity = total_perplexity * 1.0 / count_perplexities
    return average_perplexity

# Get perplexity using GPT-2
def perplexity_gpt2(file_to_score, prompt_file):

    fi = open(file_to_score, "r")

    total_perplexity = 0
    count_perplexities = 0

    prompts = [x.strip() for x in open(prompt_file, "r").readlines()]

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(device)
    stride = 512
    max_length = 1024

    for index, line in enumerate(fi):

        # Encode the prompt
        prompt = wikitext_detokenize(prompts[index])
        prompt_tokens = tokenizer.encode(prompt)
        prompt_length = len(prompt_tokens)

        # Encode the generation
        tokens_line = tokenizer.encode(wikitext_detokenize(line))

        all_tokens = prompt_tokens + tokens_line
        targets = all_tokens[:]

        # Ignore the prompt tokens in the perplexity calculation
        for i in range(prompt_length): 
            targets[i] = -100 


        # Compute average log likelihood for the generation
        lls = []
        for i in range(0, len(all_tokens), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, len(all_tokens))
            trg_len = end_loc - i    # may be different from stride on last loop
            input_ids = torch.LongTensor(all_tokens[begin_loc:end_loc]).to(device)
            target_ids = torch.LongTensor(targets[begin_loc:end_loc]).to(device)
            target_ids[:-trg_len] = -100

            # Number of tokens that get their loss computed
            # i.e., not the prompt, and not leftover context from
            # the previous segment
            count_with_loss = torch.sum(target_ids != -100).item()
            
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                log_likelihood = outputs[0] * count_with_loss

            lls.append(log_likelihood)

        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        total_perplexity += ppl.item()
        count_perplexities += 1
        logging.info("RUNNING AVERAGE PERPLEXITY AFTER LINE " + str(index) + ": " + str(total_perplexity * 1.0 / count_perplexities))

    average_perplexity = total_perplexity * 1.0 / count_perplexities
    return average_perplexity





# Update the dictionary with all ngrams of size n in line
def update_dict_specific_size(line, dictionary, ngram_size):
    # Gives a list of all ngrams of size ngram_size in line
    ngram_list = zip(*[line[i:] for i in range(ngram_size)])

    for ngram in ngram_list:
        if ngram not in dictionary:
            dictionary[ngram] = 0

# Update the dictionary with all ngrams of size (1+start_n) to max_n in line
def update_dict_size_range(line, dictionary, start_n=0, max_n=10, end_after=None):
    for i in range(start_n, max_n):
        update_dict_specific_size(line, dictionary, i+1)

# Check if all ngrams of size min_ngram_size to max_ngram_size starting at
# index start_index in train_text appear in generated_ngrams
def check_ngrams(start_index=0, min_ngram_size=1, max_ngram_size=10, train_text=None, generated_ngrams=None):

    for size in range(min_ngram_size, min(len(train_text) - start_index, max_ngram_size) + 1):
        this_ngram = tuple(train_text[start_index:start_index+size])

        if this_ngram in generated_ngrams:
            generated_ngrams[this_ngram] = 1
        else:
            # If the ngram of size k does not overlap, we know
            # for sure that no larger ngrams will overlap
            break

# Find the size of the largest ngram ending at the end
# of text that appears in ngram_set
def max_ngram_size_at_end(text=None, ngram_set=None):
    
    for size in range(1,len(text)+1):
        end_ngram = text[-1*size:]
        if tuple(end_ngram) not in ngram_set:
            return size - 1

    return len(text)


# Get a set of all the ngrams in the generated text that 
# also appear in the training set.
# gen_filename, train_filename, and prompt_filename are 
# the names of the file of generations, the training set,
# and the file of prompts
# If gen_length is provided, only the first gen_length tokens
# will be processed
# max_ngram is the maximum ngram size to consider in the first pass;
# the second pass will then fill in larger ngrams
# If add_eos is a token instead of None, that token will be added at
# the end of each line in the training file
def get_ngram_overlaps(generation_filename=None, train_filename=None, prompt_filename=None, gen_length=None, max_ngram=10, add_eos=None):

    prompt_file = open(prompt_filename, "r")
    generation_file = open(generation_filename, "r")
    train_file = open(train_filename, "r")
    

    # Get a list of all prompts and generations concatenated together, and
    # a collection of all ngrams up to length max_ngram from these concatenated
    # prompts and generations
    prompts_plus_generations = []
    prompt_plus_generation_ngrams = {}
    
    prompts = prompt_file.readlines()
    generations = generation_file.readlines()

    for prompt, generation in zip(prompts, generations):
        prompt_plus_generation = prompt.strip().split() + generation.strip().split()
        prompts_plus_generations.append(prompt_plus_generation)
        update_dict_size_range(prompt_plus_generation, prompt_plus_generation_ngrams, max_n=max_ngram)

    prompt_file.close()
    generation_file.close()


    # The largest ngram at the end of the previous line that we must carry over
    # to the current line. This is initialized as empty, since there is not yet
    # a previous line
    prev_line_carryover = []

    
    for train_index, train_line in enumerate(train_file):
        if train_index % 100000 == 0:
            logger.info("First pass over training set: Processed line #" + str(train_index))

        # A list of the tokens in this line, plus any lingering tokens from the previous line
        if add_eos is None:
            carryover_plus_line = prev_line_carryover + train_line.strip().split()
        else:
            carryover_plus_line = prev_line_carryover + train_line.strip().split() + [add_eos]

        # Check all the ngrams in this line for MAX_NGRAM-size ngrams
        for index, word in enumerate(carryover_plus_line):
            check_ngrams(start_index=index, min_ngram_size=1, max_ngram_size=max_ngram, train_text=carryover_plus_line, generated_ngrams=prompt_plus_generation_ngrams)

        # This should really only need -1*(max_ngram-1), but we're including a few extra to be safe.
        prev_line_carryover = carryover_plus_line[-1*(max_ngram + 5):]

    logger.info("done with first pass over training set")


    # The largest size of ngram overlap that we have
    # witnessed between the training set and the generations
    TRUE_MAX = max_ngram

    # Loop over all generations to search for ngrams larger than max_ngram
    # that might have appeared in training.
    # The idea here is to search for larger ngrams such that all of the ngrams
    # of size max_ngram within them have appeared in the training set, since
    # this must be satisfied for the entire larger ngram to have appeared 
    # in the training set
    for index, prompt_plus_generation in enumerate(prompts_plus_generations):
        logger.info("Processing index " + str(index) + " of the generations")
        # Minimum n-gram size that we will be updating in the Counter
        # (smaller ones have already been counted)
        min_length = max_ngram

        # Iterate over starting positions in this generation
        for start_index in range(len(prompt_plus_generation)):
            end_index = start_index + max_ngram
            increment = 0
            completed = False

            this_ngram = prompt_plus_generation[start_index:end_index]
            if tuple(this_ngram) not in prompt_plus_generation_ngrams or prompt_plus_generation_ngrams[tuple(this_ngram)] == 0:
                # Since the current ngram did not have any new overlap,
                # we can reset the minimum amount of overlap we are looking for in the next one
                min_length = max_ngram
                continue

            while not completed:
                # Check if the ngram shifted over by one has also appeared in the
                # training set
                candidate_ngram = prompt_plus_generation[start_index+increment+1:end_index+increment+1]
                
                if tuple(candidate_ngram) in prompt_plus_generation_ngrams and prompt_plus_generation_ngrams[tuple(candidate_ngram)] == 1:
                    this_ngram = candidate_ngram[:]
                    if end_index + increment + 1 > len(prompt_plus_generation):
                        # We've hit the end of the generation, so can't go any further
                        completed = True
                    else:
                        # Keep checking if there is a larger ngram that works
                        increment += 1
                else:
                    completed = True

            # This gives us a large ngram for which all of the ngrams of size
            # max_ngram within it also appear in the training set.
            complete_ngram = prompt_plus_generation[start_index:end_index+increment]

            # Updating the size of the largest (potential) overlap we might need
            # to deal with
            if len(complete_ngram) > TRUE_MAX:
                TRUE_MAX = len(complete_ngram)

            # Updating the dictionary for the generated ngrams with the ngrams in complete_ngrams
            update_dict_size_range(complete_ngram, prompt_plus_generation_ngrams, start_n=max(max_ngram, min_length), max_n=len(complete_ngram))

            # It is possible that the next complete_ngram will contain some overlap with this one. But
            # we don't want to double-count that overlap. So, for the next one, the minimum ngram size
            # that we should consider is the length of the one we just completed.
            min_length = len(complete_ngram)
    
    logger.info("done with second pass over the generations")


    train_file.close()
    train_file = open(train_filename, "r")

    prev_line_carryover = []
    
    for train_index, train_line in enumerate(train_file):
        if train_index % 10000 == 0:
            logger.info("Second pass over training set: Processed line #" + str(train_index))

        # A list of the tokens in this line, plus any lingering tokens from the previous line
        if add_eos is None:
            carryover_plus_line = prev_line_carryover + train_line.strip().split()
        else:
            carryover_plus_line = prev_line_carryover + train_line.strip().split() + [add_eos]

        # Check all the ngrams in this line for MAX_NGRAM-size ngrams
        for index, word in enumerate(carryover_plus_line):
            check_ngrams(start_index=index, min_ngram_size=max_ngram, max_ngram_size=TRUE_MAX, train_text=carryover_plus_line, generated_ngrams=prompt_plus_generation_ngrams)

        carryover_size = max_ngram_size_at_end(text=carryover_plus_line, ngram_set=prompt_plus_generation_ngrams)
        if carryover_size == 0:
            prev_line_carryover = []
        else:
            prev_line_carryover = carryover_plus_line[-1*carryover_size:]

    logger.info("done with second pass over the training set")


    return prompt_plus_generation_ngrams



# Modify the pointwise copying scores so that
# they do not factor in the prompt, only the
# generated text
def truncate_scores(scores):
    new_scores = scores[:]
    for index, score in enumerate(scores):
        if score > index + 2:
            new_scores[index] = index + 2

    return new_scores

# Input: Filename for a file containing generated
# text where each word is annotated with its pointwise
# novelty score
# Output: Lists of 2-tuples of words and their scores
def words_and_scores_from_file(filename):
    fi = open(filename, "r")

    all_words_and_scores = []

    for line in fi:
        words_and_scores = line.strip().split()

        # Removing the prompt
        words_and_scores_generated = words_and_scores[words_and_scores.index("<END_OF_PROMPT>") + 1:]

        # Split into words and scores, factoring in the
        # fact that a word might include a /
        words = ["/".join(x.split("/")[:-1]) for x in words_and_scores_generated]
        scores = [int(x.split("/")[-1]) for x in words_and_scores_generated]

        # Make the scores not factor in the prompts
        scores = truncate_scores(scores)

        all_words_and_scores.append(list(zip(words, scores)))

    return all_words_and_scores


# Get various types of average pointwise novelty scores
def avg_pointwise_novelty(all_words_and_scores):
    total_words = 0
    total_pointwise_scores = 0
    total_inverse_pointwise_scores = 0
    total_log2_pointwise_scores = 0
    total_ln_pointwise_scores = 0
    total_truncated5_pointwise_scores = 0

    for words_and_scores in all_words_and_scores:
        for word, score in words_and_scores:
            total_words += 1
            total_pointwise_scores += score
            total_inverse_pointwise_scores += 1.0/score
            total_log2_pointwise_scores += math.log(score, 2)
            total_ln_pointwise_scores += math.log(score)
            total_truncated5_pointwise_scores += min(score, 5)
    
    pointwise_score = total_pointwise_scores * 1.0 / total_words
    inverse_pointwise_score = total_inverse_pointwise_scores * 1.0 / total_words
    log2_pointwise_score = total_log2_pointwise_scores * 1.0 / total_words
    ln_pointwise_score = total_ln_pointwise_scores * 1.0 / total_words
    truncated5_pointwise_score = total_truncated5_pointwise_scores * 1.0 / total_words

    return pointwise_score, inverse_pointwise_score, log2_pointwise_score, ln_pointwise_score, truncated5_pointwise_score

# Get the average pointwise copying score by position
# within the generated text
def avg_pointwise_novelty_by_position(all_words_and_scores):
    max_length = 0
    for words_and_scores in all_words_and_scores:
        if len(words_and_scores) > max_length:
            max_length = len(words_and_scores)

    totals_and_counts_by_position = [[0,0] for _ in range(max_length)]
    truncated_totals_and_counts_by_position = [[0,0] for _ in range(max_length)]

    for words_and_scores in all_words_and_scores:
        for index, (word, score) in enumerate(words_and_scores):
            totals_and_counts_by_position[index][0] += score
            totals_and_counts_by_position[index][1] += 1

            truncated_totals_and_counts_by_position[index][0] += min(score, 10)
            truncated_totals_and_counts_by_position[index][1] += 1

    averages_by_position = []
    for total, count in totals_and_counts_by_position:
        average = total*1.0 / count
        averages_by_position.append(average)
    
 
    truncated_averages_by_position = []
    for total, count in truncated_totals_and_counts_by_position:
        average = total*1.0 / count
        truncated_averages_by_position.append(average)
    
        
    binned_averages = []
    total = 0
    count = 0
    for index, average in enumerate(averages_by_position):
        total += average
        count += 1

        if (index + 1) % 100 == 0:
            binned_averages.append(total*1.0/count)
            total = 0
            count = 0

    if count != 0:
        binned_averages.append(total*1.0/count)

      
    truncated_binned_averages = []
    total = 0
    count = 0
    for index, average in enumerate(truncated_averages_by_position):
        
        # Ignore indices 0 through 8, as they cannot possibly have
        # a score of 10, and we don't want that to distort the results
        # due to ceiling effects
        if index < 9:
            continue

        total += average
        count += 1

        if (index + 1) % 100 == 0:
            truncated_binned_averages.append(total*1.0/count)
            total = 0
            count = 0

    if count != 0:
        truncated_binned_averages.append(total*1.0/count)

    return averages_by_position, binned_averages, truncated_averages_by_position, truncated_binned_averages


# For a given generation length, update the
# maximum overlap counts that we could get
# from this length
def add_length(length, maximum_overlap_counts):
    for size in range(1,length+1):
        # Add this ngram size to the dictionary if needed
        if size not in maximum_overlap_counts:
            maximum_overlap_counts[size] = 0

        # For a generation of with length 'length', there
        # are (length - size + 1) ngrams of size 'size' within
        # that generation. 
        maximum_overlap_counts[size] += length - size + 1

# For a given pointwise score, update the counts of overlapping
# ngrams.
def add_score(score, overlap_counts):
    # Due to the way the scores are computed, the 
    # size of the overlapping ngram is the score minus 1
    ngram_size = score - 1

    # We know that not only this ngram, but also all smaller-size
    # ngrams, are overlaps
    for i in range(1, ngram_size+1):
        # Add this ngram size to the dictionary if needed
        if i not in overlap_counts:
            overlap_counts[i] = 0
        overlap_counts[i] += 1



def ngram_overlap_from_pointwise(all_words_and_scores):
    # A dict where keys are ngram sizes, and values are the
    # number of ngrams that overlap with the training set
    overlap_counts = {}

    # A dict where keys are ngram sizes, and values are the
    # number of ngrams of that size that we have in our generated
    # text (i.e., the maximum possible overlap we could have)
    maximum_overlap_counts = {}

    for words_and_scores in all_words_and_scores:
        # Update the maximum_overlap_counts
        add_length(len(words_and_scores), maximum_overlap_counts)

        # Update the overlap_counts
        for word, score in words_and_scores:
            add_score(score, overlap_counts)

    size_list = []
    proportion_list = []

    size = 1
    while size in overlap_counts:
        size_list.append(size)
        proportion_list.append(overlap_counts[size]*1.0/maximum_overlap_counts[size])
        size += 1

    return size_list, proportion_list


# Returns a list of length 'count' of novel bigrams
# and their contexts (10 words to the left and right)
# Only bigrams that have 10 words of context to the left
# and right are considered
def novel_bigram_examples(all_words_and_scores, count):
    bigrams_in_context = []

    for words_and_scores in all_words_and_scores:
        for index, (word, score) in enumerate(words_and_scores):
            if score == 2 and index >= 11 and index <= len(words_and_scores) - 11:
                preceding = words_and_scores[index - 11:index - 1]
                preceding = [x[0] for x in preceding]
                bigram = [words_and_scores[index-1][0], word]
                following = words_and_scores[index + 1:index + 11]
                following = [x[0] for x in following]
                overall = " ".join(preceding) + " " + "***" + " ".join(bigram) + "***" + " " + " ".join(following)

                bigrams_in_context.append(overall)
    
    shuffle(bigrams_in_context)
    return bigrams_in_context[:count]


# Returns a list of length 'count' of supercopying
# (i.e., copying of 100 words or more from the training set)
# and their contexts (10 words to the left and right, if available)
# Uses a maximal definition where, if a supercopied passage
# is length 137, it will return that - not a substring of
# length 135 or 100.
def supercopying_examples(all_words_and_scores, count):
    supercopying_in_context = []
    longest_example = ""
    length_longest = 0

    for words_and_scores in all_words_and_scores:
        for index, (word, score) in enumerate(words_and_scores):
            if score >= 101:
                if index < len(words_and_scores) - 1:
                    if score < words_and_scores[index+1][1]:
                        continue
                preceding = words_and_scores[max(index - (score+8),0):max(index - (score-2),0)]
                preceding = [x[0] for x in preceding]
                supercopying = words_and_scores[index-(score-2):index+1]
                supercopying = [x[0] for x in supercopying]
                following = words_and_scores[min(index + 1,len(words_and_scores)):index + 11]
                following = [x[0] for x in following]
                overall = " ".join(preceding) + " " + "***" + " ".join(supercopying) + "***" + " " + " ".join(following)

                if len(supercopying) > length_longest:
                    longest_example = " ".join(supercopying)
                    length_longest = len(supercopying)

                supercopying_in_context.append(overall)
    
    shuffle(supercopying_in_context)
    return supercopying_in_context[:count], longest_example, length_longest



# Returns a list of all instances of supercopying
# (i.e., copying of 100 words or more from the training set)
def all_supercopies(all_words_and_scores):
    supercopies = []

    for words_and_scores in all_words_and_scores:
        for index, (word, score) in enumerate(words_and_scores):
            if score >= 101:
                supercopy = words_and_scores[index-99:index+1]
                supercopy = [x[0] for x in supercopy]

                supercopies.append(supercopy)
    
    return supercopies



# Return a list of all full supercopied passages: not split
# up into 100-grams, but instead the whole passage
def all_supercopies_full(all_words_and_scores):
    supercopies = []

    for words_and_scores in all_words_and_scores:
        for index, (word, score) in enumerate(words_and_scores):
            if score >= 101:
                if index < len(words_and_scores) - 1:
                    if score < words_and_scores[index+1][1]:
                        continue

                full_passage_start_index = index-(score-2)
                full_passage_end_index = index+1

                supercopy = words_and_scores[full_passage_start_index:full_passage_end_index]
                supercopy = [x[0] for x in supercopy]

                supercopies.append(supercopy)
    
    return supercopies

# For each full supercopying example, find the most-repeated 100-gram 
# within it, and report the counts of those 100-grams
def count_supercopy_max(all_supercopying_examples, supercopy_count_list, all_supercopying_examples_full):
    supercopy2count = {}

    for supercopy, count in zip(all_supercopying_examples, supercopy_count_list):
        supercopy2count[tuple(supercopy)] = count

    max_overlaps = []
    for full_supercopy in all_supercopying_examples_full:
        max_overlap = 0
        for start_index in range(len(full_supercopy) - 99):
            hundredgram = full_supercopy[start_index:start_index+100]

            count_hundredgram = supercopy2count[tuple(hundredgram)]
            if count_hundredgram > max_overlap:
                max_overlap = count_hundredgram

        max_overlaps.append(max_overlap)

    sum_overlaps = 0
    count_overlaps = 0

    for max_overlap in max_overlaps:
        sum_overlaps += max_overlap
        count_overlaps += 1

    if count_overlaps == 0:
        average_overlap = 0
    else:
        average_overlap = sum_overlaps * 1.0 / count_overlaps

    return average_overlap, max_overlaps



# Assumes all the ngrams have the same length
# For each ngram in ngram_list, counts how many times
# it occurs in the file 'filename'
def count_occurrences_in_file(ngram_list, filename, eos=None, standardize=False):
    if ngram_list == []:
        return 0, [], 0, "", 0

    ngram_length = len(ngram_list[0])
    ngram_dict = {}
    for ngram in ngram_list:
        ngram_dict[tuple(ngram)] = 0

    fi = open(filename, "r")
    current_segment = []

    for index, line in enumerate(fi):
        if index % 100000 == 0:
            logging.info(str(index))
        words = line.strip().split()

        if eos is not None:
            words = words + [eos]

        current_segment = current_segment + words
        if standardize:
            current_segment = format_text_for_checking_overlap(current_segment)

        for i in range(len(current_segment)):
            if i + ngram_length > len(current_segment):
                current_segment = current_segment[max(0, len(current_segment) - ngram_length + 1):]
                break
            else:
                current_ngram = tuple(current_segment[i:i+ngram_length])
                if current_ngram in ngram_dict:
                    ngram_dict[current_ngram] += 1


    # Compute average occurrences
    total_overlap = 0
    count_nonzero = 0
    count_overlap = 0

    # List of all overlap values
    overlap_list = []

    max_overlap = 0
    max_overlap_ngram = None

    for key in ngram_list:
        this_overlap = ngram_dict[tuple(key)]
        total_overlap += this_overlap
        if this_overlap != 0:
            count_nonzero += 1
        count_overlap += 1

        overlap_list.append(this_overlap)

        if this_overlap > max_overlap:
            max_overlap = this_overlap 
            max_overlap_ngram = tuple(key)

    average_overlap = total_overlap*1.0 / count_overlap
    proportion_nonzero = count_nonzero*1.0 / count_overlap

    return average_overlap, overlap_list, max_overlap, max_overlap_ngram, proportion_nonzero

# Returns the number of words in a file
def words_in_file(filename, eos=None):
    word_count = 0

    fi = open(filename, "r")
    for line in fi:
        words = line.strip().split()
        word_count += len(words)
        if eos is not None:
            word_count += 1

    return word_count

# Given a maximum index, returns 'count' indices
# selected randomly without replacement 
def start_indices(max_index, count, length=100):

    indices = random.sample(range(max_index-length), count)

    return indices

# Returns 'count' ngrams of length 'length' selected randomly
# from 'filename'
def random_ngrams(filename, count, length=100, eos=None):
    word_count = words_in_file(filename, eos=eos)
    start_list = sorted(start_indices(word_count, count, length=length))
    ngram_list = []

    fi = open(filename, "r")

    current_start_index = 0
    current_segment = []

    for index, line in enumerate(fi):
        if index % 100000 == 0:
            logging.info("Processing: " +  str(index))
        new_words = line.strip().split()
        current_segment = current_segment + new_words
        
        if eos is not None:
            current_segment = current_segment + [eos]

        if start_list[0] <= current_start_index + len(current_segment):
            if start_list[0] + length <= current_start_index + len(current_segment):
                new_ngram = current_segment[start_list[0]-current_start_index:start_list[0]-current_start_index+length]
                ngram_list.append(new_ngram)
                start_list = start_list[1:]
                if start_list == []:
                    break

        else:
            current_start_index += len(current_segment)
            current_segment = []


    return ngram_list

# A way to standardize text for comparing across corpora
# The input should be longer than you want, to allow for the 
# deletion of words making it shorter.
# Any word that does not fully consist of letters will be
# deleted. The remaining words will be lowercased.
def format_text_for_checking_overlap(text, length=None):
    done = False
    
    while not done:
        new_text = []

        for word in text:
            if word.isalpha() or word == "<eos>":
                new_text.append(word.lower())

        if length is not None:
            if len(new_text) < length:
                logging.info("Input text is too short!")
                return None
            else:
                candidate = new_text[:length]
                if "<eos>" in candidate:
                    text = text[length:]
                    if len(text) < length:
                        logging.info("Too many <eos> tokens!")
                        return None
                else:
                    return candidate
        else:
            return new_text






