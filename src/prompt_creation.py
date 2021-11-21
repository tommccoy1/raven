import random
import argparse
from transformers import GPT2Tokenizer
import numpy as np
import logging

# Turns a source file into a list of tokens
def source_to_list(source_file, newline_token=None, token_before_prompt=None, filetype=None):
    if filetype in ["wikitext"]:
        fi = open(source_file, "r")
    
        token_list = []

        # We assume that the prompt can begin at the start of the file, even if the
        # token that precedes possible prompts does not appear there explicitly
        if token_before_prompt is not None:
            token_list.append(token_before_prompt)

        # Read the file into a list, adding newline tokens if specified
        if newline_token is not None:
            for line in fi:
                token_list = token_list + line.strip().split() + [newline_token]
        else:
            for line in fi:
                token_list = token_list + line.strip().split()

        # Create a list of all indices after which a prompt could start
        # If prompts don't have to begin after a particular token, this 
        # is all indices; otherwise, it's the indices followed by that
        # token (token_before_prompt)
        if token_before_prompt is None:
            start_indices = list(range(len(token_list)))
        else:
            start_indices = []
            for index, elt in enumerate(token_list):
                if elt == token_before_prompt:
                    start_indices.append(index)
    
        fi.close()
    
    elif filetype in ["webtext"]:
        # This is a .npy (numpy) file
        token_list = np.load(source_file)

        # We do not add token_before_prompt at the start
        # because the file already starts with it

        if token_before_prompt is None:
            start_indices = list(range(len(token_list)))
        else:
            # Each token is an int, not a string
            token_before_prompt = int(token_before_prompt)
            
            start_indices = []

            for start_index, elt in enumerate(token_list):
                if start_index % 1000000 == 0:
                    logger.info("Processed position " + str(start_index) + " out of " + str(len(token_list)))
                if token_list[start_index] == token_before_prompt:
                    start_indices.append(start_index)


    return token_list, start_indices

# Implements restrictions on the types of prompt + continuation
# sequences that we want to allow for given types of source files
def valid_prompt_and_continuation(sequence, filetype="general"):
    if filetype == "wikitext":
        # Don't want prompt to start with eos
        if sequence[0] == "<eos>":
            return False

        return True

    elif filetype == "webtext":
        # Don't want to span more than one document
        if 50256 in sequence:
            return False

        return True

    else:
        return True


if __name__ == "__main__":

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nprompts_per_file", type=int, default=1000, help="Number of prompts per generated file of prompts")
    parser.add_argument("--nfiles", type=int, default=1, help="Number of files of prompts to generate")
    parser.add_argument("--prompt_length", type=int, default=512, help="Number of tokens in each prompt")
    parser.add_argument("--continuation_length", type=int, default=1010, help="Number of tokens in the continuation after the prompt (this will be how long the generations are planned to be). It's safest to overestimate this, as you can always trim it down afterwards.")
    parser.add_argument("--source_file", type=str, default=None, help="File from which prompts will be generated")
    parser.add_argument("--token_before_prompt", type=str, default=None, help="Token after which a prompt may start; e.g., could be <eos> if you want the prompt to start at the beginning of a sentence. Leave as None if you want the prompts to start anywhere")
    parser.add_argument("--newline_token", type=str, default=None, help="Token to add for each newline in source_file. Leave as None if you don't want to add a newline character")
    parser.add_argument("--filetype", type=str, default=None, help="File type of source_file (e.g., wikitext, webtext)")
    parser.add_argument("--output_prefix", type=str, default=None, help="Prefix for the files of prompts and continuations")
    args = parser.parse_args()

    if args.filetype == "webtext":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Turn the source file into a list of tokens, and generate
    # a list of the indices in that list after which a prompt 
    # could begin
    token_list, start_indices = source_to_list(args.source_file, newline_token=args.newline_token, token_before_prompt=args.token_before_prompt, filetype=args.filetype)


    # We will be generating nfiles
    for file_number in range(args.nfiles):
        fo_prompts = open(args.output_prefix + "_prompts_length" + str(args.prompt_length) + "_" + str(file_number + 1) + "of" + str(args.nfiles) + ".txt", "w", encoding="utf-8")
        fo_continuations = open(args.output_prefix + "_continuations_length" + str(args.prompt_length) + "_" + str(file_number + 1) + "of" + str(args.nfiles) + ".txt", "w", encoding="utf-8")

        # For each file, we generate nprompts_per_file prompts
        for prompt_number in range(args.nprompts_per_file):
            print(prompt_number)
            satisfied = False # Whether we have successfully generated a prompt yet in this iterations

            while not satisfied:
                start_index = random.choice(start_indices)
       
                if start_index + args.prompt_length + 1 + args.continuation_length > len(token_list):
                    # We don't want to run over the length of the source file
                    satisfied = False
                else:
                    # Gives the potential prompt and continuation. 1 is added because
                    # we want the prompt to start *after* args.token_before_prompt.
                    full_sequence = token_list[start_index + 1: start_index + 1 + args.prompt_length + args.continuation_length]
                    prompt_end_index = start_index + args.prompt_length + 1

                    if args.filetype in ["wikitext"]:
                        satisfied = valid_prompt_and_continuation(full_sequence, filetype=args.filetype)
                    elif args.filetype in ["webtext"]:
                        
                        # Make sure that the prompt ends in a complete word
                        if valid_prompt_and_continuation(full_sequence, filetype=args.filetype):
                            good_ending = False
                            
                            if args.prompt_length == 0:
                                good_ending = True
                                satisfied = True

                            while not good_ending:
                                next_token = tokenizer.decode([token_list[prompt_end_index]])
                                if next_token[0] == " ":
                                    good_ending = True
                                    satisfied = True
                                else:
                                    prompt_end_index += 1
                                    
                                    # We don't want to extend the prompt by too many tokens; so,
                                    # if that would be necessary to make it end on a whole word, 
                                    # we just discard this potential prompt
                                    if prompt_end_index - start_index - args.prompt_length - 1 > 10:
                                        satisfied = False
                                        break

                        else:
                            satisfied = False

            # For a prompt of length 0, we use just the EOS token
            if args.prompt_length == 0:
                prompt = [args.token_before_prompt]
                continuation = token_list[start_index + 1: start_index + 1 + args.prompt_length + args.continuation_length]
            else:
                prompt = token_list[start_index + 1: prompt_end_index]
                continuation = token_list[prompt_end_index: start_index + 1 + args.prompt_length + args.continuation_length]

            fo_prompts.write(" ".join([str(token) for token in prompt]) + "\n")
            fo_continuations.write(" ".join([str(token) for token in continuation]) + "\n")

            # Make sure that none of the chosen prompts are from
            # the same position as each other by removing the 
            # start index that we just used
            new_start_indices = []
            for index in start_indices:
                if index == start_index:
                    pass
                else:
                    new_start_indices.append(index)

            start_indices = new_start_indices[:]




