
import subprocess
import argparse
import logging

# Update the dictionary with all ngrams of size n in line
def update_dict_specific_size(line, dictionary, ngram_size):
    ngram_list = zip(*[line[i:] for i in range(ngram_size)])
    for ngram in ngram_list:
        if ngram not in dictionary:
            dictionary[ngram] = 0

# Update the dictionary with all ngrams of size (1+start_n) to max_n in line
def update_dict_size_range(line, dictionary, start_n=0, max_n=10, end_after=None):
    for i in range(start_n, max_n):
        update_dict_specific_size(line, dictionary, i+1)


parser = argparse.ArgumentParser()
parser.add_argument("--generation_file", help="file of generations", type=str, default=None)
parser.add_argument("--prompt_file", help="file of prompts", type=str, default=None)
parser.add_argument("--training_file", help="training file to compare to", type=str, default=None)
args = parser.parse_args()

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

overlap_ngrams = {}

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
    prompt_plus_generation = prompt.strip().split() + generation.strip().split()
    prompts_plus_generations.append(prompt_plus_generation)



fo_scored = open(args.generation_file + ".pointwise_grep", "w")
for prompt_index, prompt_plus_generation in enumerate(prompts_plus_generations):
    logging.info("PROMPT NUMBER: " + str(prompt_index))
    scores = [0 for _ in range(len(prompt_plus_generation))]
    latest_end_index = 0   
    for start_index in range(len(prompt_plus_generation)):
        logging.info("START INDEX: " + str(start_index))

        for end_index in range(latest_end_index, len(prompt_plus_generation) + 1):
            logging.info("END INDEX: " + str(end_index))
            if end_index == start_index:
                continue
            #print(prompt_index)
            
            tuple_subpart = tuple(prompt_plus_generation[start_index:end_index])

            if tuple_subpart in overlap_ngrams:
                subpart_in_training = True

            else:
                subpart = " ".join(prompt_plus_generation[start_index:end_index])
                logging.info(subpart)

                #print(subpart)
                #print("")

                #bash_command = 'LC_ALL=C grep "' + subpart + '" ' + args.training_file + ' -m 1 -F -q'
                bash_command = 'LC_ALL=C grep -m 1 -F -q -- "' + subpart + '" ' + args.training_file


                exit_status = subprocess.call(['bash','-c', bash_command], stderr=subprocess.STDOUT)
                if exit_status == 0:
                    subpart_in_training = True
                    #print("Got one!", subpart)
                    update_dict_size_range(tuple_subpart, overlap_ngrams, max_n=len(tuple_subpart)+10)
                else:
                    subpart_in_training = False



            if not subpart_in_training:
                stopping_index = end_index - 1

                for index in range(latest_end_index, stopping_index):
                    if latest_end_index + 1 == end_index:
                        scores[index] = 1
                    else:
                        scores[index] = index - start_index + 2

                latest_end_index = stopping_index

                break

            elif end_index == len(prompt_plus_generation): # - 1:
                
                for index in range(latest_end_index, end_index):
                    scores[index] = index - start_index + 2

                latest_end_index = end_index + 1
                
                break

           

    prompt = prompts[prompt_index].strip().split()
    generation = generations[prompt_index].strip().split()

    prompt_scores = scores[:len(prompt)]
    generation_scores = scores[len(prompt):]

    annotated_prompt = " ".join([word + "/" + str(score) for (word, score) in zip(prompt, prompt_scores)])
    annotated_generation = " ".join([word + "/" + str(score) for (word, score) in zip(generation, generation_scores)])
    
    fo_scored.write(annotated_prompt + " <END_OF_PROMPT> " + annotated_generation + "\n")



