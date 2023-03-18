import json
import collections
import argparse

from analyses import *


parser = argparse.ArgumentParser()
parser.add_argument("--generation_file", help="file of generations", type=str, default=None)
parser.add_argument("--prompt_file", help="file of prompts", type=str, default=None)
parser.add_argument("--training_file", help="training file to compare to", type=str, default=None)
parser.add_argument("--max_ngram", help="Maximum n-gram overlap size to start with (but the code will adaptively expand beyond this as needed)", type=int, default=10)
parser.add_argument("--gen_length", help="Length of generated texts; if specified, this will truncate generations to that length", type=int, default=None)
parser.add_argument("--n_gens", help="Number of generated texts", type=int, default=1000)
parser.add_argument("--eos", help="EOS token to add at the end of every training line, if any", type=str, default=None)
args = parser.parse_args()


full_training_string = []
training_file = open(args.training_file, "r")
for line in training_file:
    if args.eos is None:
        full_training_string = full_training_string + line.strip().split()
    else:
        full_training_string = full_training_string + line.strip().split() + [args.eos]

print("have training string")

train_ngrams = set()
for start_index in range(len(full_training_string)):
    print(start_index, len(full_training_string))
    for end_index in range(start_index+1, len(full_training_string) + 1):
        train_ngrams.add(tuple(full_training_string[start_index:end_index]))



prompt_file = open(args.prompt_file, "r")
generation_file = open(args.generation_file, "r")

# Get a list of all prompts and generations concatenated together, and
# a collection of all ngrams up to length max_ngram from these concatenated
# prompts and generations
prompts_plus_generations = []

prompts = prompt_file.readlines()
generations = generation_file.readlines()

for prompt, generation in zip(prompts, generations):
    # Storing prompt and generation as a 2-tuple
    prompt_plus_generation = (prompt.strip().split(), generation.strip().split())
    prompts_plus_generations.append(prompt_plus_generation)

prompt_file.close()
generation_file.close()


scored_prompts_and_generations = []

for position, prompt_and_generation in enumerate(prompts_plus_generations):
    if position % 10 == 0:
        print("Scoring:", position)

    scores = []
    
    prompt = prompt_and_generation[0]
    generation = prompt_and_generation[1]
    prompt_plus_generation = prompt + generation
    for index, token in enumerate(prompt_plus_generation):
        start_point = index
        end_point = index + 1
        done = False

        while not done:
            ngram = tuple(prompt_plus_generation[start_point:end_point])
            if ngram not in train_ngrams or start_point == 0:
                done = True
            if ngram in train_ngrams:
                start_point -= 1

        score = end_point - start_point
        scores.append(score)

    prompt_scores = scores[:len(prompt)]
    generation_scores = scores[len(prompt):]
    if len(prompt_scores) != len(prompt) or len(generation_scores) != len(generation):
        print("ERROR! Wrong score lengths")

    scored_prompts_and_generations.append(((prompt, prompt_scores), (generation, generation_scores)))

fo_scored = open(args.generation_file + ".pointwise_exhaustive", "w")
for index, scored_prompt_and_generation in enumerate(scored_prompts_and_generations):
    prompt = scored_prompt_and_generation[0][0]
    prompt_scores = scored_prompt_and_generation[0][1]
    generation = scored_prompt_and_generation[1][0]
    generation_scores = scored_prompt_and_generation[1][1]

    annotated_prompt = " ".join([word + "/" + str(score) for (word, score) in zip(prompt, prompt_scores)])
    annotated_generation = " ".join([word + "/" + str(score) for (word, score) in zip(generation, generation_scores)])

    fo_scored.write(annotated_prompt + " <END_OF_PROMPT> " + annotated_generation + "\n") 
    
            



