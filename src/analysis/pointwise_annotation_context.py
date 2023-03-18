import json
import collections
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--generation_file", help="file of generations", type=str, default=None)
parser.add_argument("--prompt_file", help="file of prompts", type=str, default=None)
parser.add_argument("--to_score", help="what to compute copying from: 'prompt' for the prompt, 'previous_generation' for previously generated text, 'context' for the concatenation of prompt and previously generated text", type=str, default=None)
args = parser.parse_args()


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

for index, prompt_and_generation in enumerate(prompts_plus_generations):
    prompt = prompt_and_generation[0]
    generation = prompt_and_generation[1]
    prompt_plus_generation = prompt + generation

    scores = [0 for _ in range(len(prompt_plus_generation))]

    for end_index in range(len(prompt_plus_generation)):
        if args.to_score == "context":
            context = prompt_plus_generation[:end_index]
        elif args.to_score == "prompt":
            context = prompt
        elif args.to_score == "previous_generation":
            if end_index <= len(prompt):
                context = []
            else:
                context = generation[:end_index - len(prompt)]

        context_string = " " + " ".join(context) + " "

        if end_index < len(prompt):
            scores[end_index] = end_index + 2
            continue

        scored = False
        for length in range(end_index):
            excerpt = prompt_plus_generation[end_index-length:end_index+1]
            excerpt = " " + " ".join(excerpt) + " "
            if excerpt not in context_string:
                scores[end_index] = length+1
                scored = True
                break
        if not scored:
            scores[end_index] = end_index+2


    scored_prompts_and_generations.append((prompt_and_generation, (scores[:len(prompt)], scores[len(prompt):])))

fo_scored = open(args.generation_file + "." + args.to_score + "_pointwise", "w")


for index, scored_prompt_and_generation in enumerate(scored_prompts_and_generations):
    prompt = scored_prompt_and_generation[0][0]
    prompt_scores = scored_prompt_and_generation[1][0]
    generation = scored_prompt_and_generation[0][1]
    generation_scores = scored_prompt_and_generation[1][1]

    annotated_prompt = " ".join([word + "/" + str(score) for (word, score) in zip(prompt, prompt_scores)])
    annotated_generation = " ".join([word + "/" + str(score) for (word, score) in zip(generation, generation_scores)])

    fo_scored.write(annotated_prompt + " <END_OF_PROMPT> " + annotated_generation + "\n") 
    
            



