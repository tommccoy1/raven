
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import T5Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None, help="File to sentence tokenize")
parser.add_argument("--max_word_count", type=int, default=1000, help="Max number of words per sentence")
parser.add_argument("--max_subword_count", type=int, default=None, help="Max number of T5 subwords per sentence")
parser.add_argument("--max_char_count", type=int, default=5000, help="Max number of characters per sentence")
parser.add_argument("--max_nonalpha_count", type=int, default=100, help="Max number of non-letters per sentence")
parser.add_argument("--generation", action="store_true", help="Whether the input file is a text generation (and therefore comes with a file of prompts and needs to have line indices stored)")
parser.add_argument("--space_before_generation", action="store_true", help="Always add a space between the prompt and the generation") 
parser.add_argument("--prompt_filename", type=str, default=None, help="File of prompts")
parser.add_argument("--newline", type=str, default=None, help="Newline token")
parser.add_argument("--unk", type=str, default=None, help="unk token")
args = parser.parse_args()


tokenizer = T5Tokenizer.from_pretrained('t5-small')

fi = open(args.filename, "r")

fo = open(args.filename + ".sentences", "w")
fo_split = open(args.filename + ".sentences_split", "w")

max_word_count = args.max_word_count
max_char_count = args.max_char_count
max_nonalpha_count = args.max_nonalpha_count
max_subword_count = args.max_subword_count

if max_subword_count is None:
    max_subword_count = max_word_count*100000

# Return the number of characters in the sentence that
# are neither letters nor spaces (i.e., that are 
# punctuation or numbers)
def count_nonalpha(sentence):
    count = 0
    for char in sentence:
        if not char.isalpha() and char != " ":
            count += 1

    return count

def split_overlong(sentence):

    words = sentence.split()
    new_sentences = []

    current_new_sentence = []
    for word in words:
        current_new_sentence.append(word)
        sentence_len = len(word_tokenize(" ".join(current_new_sentence)))
        subwords_len = len(tokenizer(" ".join(current_new_sentence))["input_ids"])
        if sentence_len >= max_word_count or len(" ".join(current_new_sentence)) >= max_char_count or count_nonalpha(" ".join(current_new_sentence)) >= max_nonalpha_count or subwords_len >= max_subword_count:
            new_sentences.append(" ".join(current_new_sentence))
            current_new_sentence = []

    new_sentences.append(" ".join(current_new_sentence))
    
    redo = False
    for new_sentence in new_sentences:
        if len(word_tokenize(new_sentence)) > max_word_count + 10 or len(new_sentence) > max_char_count + 10 or count_nonalpha(new_sentence) > max_nonalpha_count + 10 or len(tokenizer(new_sentence)["input_ids"]) > max_subword_count + 10:
            redo = True

    if redo:
        words = word_tokenize(sentence)
        new_sentences = []

        current_new_sentence = []
        for word in words:
            current_new_sentence.append(word)
            subwords_len = len(tokenizer(" ".join(current_new_sentence))["input_ids"])
            if len(current_new_sentence) >= max_word_count or len(" ".join(current_new_sentence)) >= max_char_count or count_nonalpha(" ".join(current_new_sentence)) >= max_nonalpha_count or subwords_len >= max_subword_count:
                new_sentences.append(" ".join(current_new_sentence))
                current_new_sentence = []

        new_sentences.append(" ".join(current_new_sentence))

    redo_redo = False
    for new_sentence in new_sentences:
        if len(word_tokenize(new_sentence)) > max_word_count + 10 or len(new_sentence) > max_char_count + 10 or count_nonalpha(new_sentence) > max_nonalpha_count + 10 or len(tokenizer(new_sentence)["input_ids"]) > max_subword_count + 10:
            redo_redo = True

    if redo_redo:
        new_sentences = []

        current_new_sentence = ""
        for char in sentence:
            current_new_sentence = current_new_sentence + char
            subwords_len = len(tokenizer(current_new_sentence)["input_ids"])
            if len(current_new_sentence) >= max_char_count or count_nonalpha(current_new_sentence) >= max_nonalpha_count or subwords_len >= max_subword_count:
                new_sentences.append(current_new_sentence)
                current_new_sentence = ""

        new_sentences.append(current_new_sentence)

    if redo_redo:
        return new_sentences, "char-level"
    elif redo:
        return new_sentences, "word-tokenized"
    else:
        return new_sentences, "space-delimited"

count_overlong = 0
count_char_level = 0
count_word_tokenized = 0
count_space_delimited = 0
count_sentences = 0


def sentences_trim(sentences):
    count_overlong_internal = 0
    count_char_level_internal = 0
    count_word_tokenized_internal = 0
    count_space_delimited_internal = 0
    count_sentences_internal = 0

    sentences_to_return = []
    sentences_split_to_return = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        subwords = tokenizer(sentence)["input_ids"]
        if len(words) <= max_word_count and len(sentence) <= max_char_count and count_nonalpha(sentence) <= max_nonalpha_count and len(subwords) <= max_subword_count:
            sentences_to_return.append(sentence)
        else:
            sentences_split_to_return.append(sentence)
            split_sentences, category = split_overlong(sentence)
            for split_sentence in split_sentences:
                sentences_to_return.append(split_sentence)

            count_overlong_internal += 1
            if category == "char-level":
                count_char_level_internal += 1
            if category == "word-tokenized":
                count_word_tokenized_internal += 1
            if category == "space-delimited":
                count_space_delimited_internal += 1

        count_sentences_internal += 1

    return sentences_to_return, sentences_split_to_return, count_overlong_internal, count_char_level_internal, count_word_tokenized_internal, count_space_delimited_internal, count_sentences_internal



if args.generation:
    if args.prompt_filename is not None:
        fi_prompt = open(args.prompt_filename, "r")
        prompts = fi_prompt.readlines()

    for line_index, generation in enumerate(fi):
        print(line_index)

        if args.prompt_filename is None:
            prompt = ""
        else:
            prompt = prompts[line_index]

        # Whether the generation starts with a space
        generation_starts_with_space = (generation[0] == " ")

        prompt = prompt.strip()
        generation = generation.strip()

        # Remove brackets from unk tokens
        if args.unk is not None:
            prompt = prompt.replace(args.unk, "unk")
            generation = generation.replace(args.unk, "unk")

        # Change the newline characters into actual newlines
        prompt = prompt.replace(args.newline, "\n").replace(" \n", "\n").replace("\n ", "\n")
        generation = generation.replace(args.newline, "\n").replace(" \n", "\n").replace("\n ", "\n")

        # Split the generation into sentences
        generation = generation.split("\n")

        sentences_generation = []
        for paragraph_generation in generation:
            sentences_generation = sentences_generation + sent_tokenize(paragraph_generation.strip())

        # Add the first generation sentence to the prompt
        if generation_starts_with_space or args.space_before_generation:
            prompt = prompt + " " + sentences_generation[0].strip()
        else:
            prompt = prompt + sentences_generation[0].strip()

        # Ignore the first and last generation sentences, since they might not be complete sentences
        sentences_generation = sentences_generation[1:-1]
        
        # Split the prompt into sentences
        prompt = prompt.split("\n")

        sentences_prompt = []
        for paragraph_prompt in prompt:
            sentences_prompt = sentences_prompt + sent_tokenize(paragraph_prompt.strip())

        # Split up any extra-long senttences
        sentences_prompt_to_return, sentences_split_prompt_to_return, count_overlong_internal, count_char_level_internal, count_word_tokenized_internal, count_space_delimited_internal, count_sentences_internal = sentences_trim(sentences_prompt)
        count_overlong += count_overlong_internal
        count_char_level += count_char_level_internal
        count_word_tokenized += count_word_tokenized_internal
        count_space_delimited += count_space_delimited_internal
        count_sentences += count_sentences_internal

        sentences_generation_to_return, sentences_split_generation_to_return, count_overlong_internal, count_char_level_internal, count_word_tokenized_internal, count_space_delimited_internal, count_sentences_internal = sentences_trim(sentences_generation)
        count_overlong += count_overlong_internal
        count_char_level += count_char_level_internal
        count_word_tokenized += count_word_tokenized_internal
        count_space_delimited += count_space_delimited_internal
        count_sentences += count_sentences_internal

        # Index of the first generation sentence in the prompt + generation
        first_generation_index = len(sentences_prompt_to_return)

        # Print the sentences
        all_sentences_to_print = sentences_prompt_to_return + sentences_generation_to_return
        all_sentences_split = sentences_split_prompt_to_return + sentences_split_generation_to_return

        for sentence_index, sentence in enumerate(all_sentences_to_print):
            # Each sentence is followed by the index of the line it is from, the index of the
            # first generation sentence in that line, and the index of this sentence
            # within the line
            fo.write(sentence + "\t" + str(line_index) + "\t" + str(first_generation_index) + "\t" + str(sentence_index) + "\n")

        for sentence in all_sentences_split:
            fo_split.write(sentence + "\n")


else:
    for line in fi:

        line = line.split("\n")

        # Tokenize the generated sentences
        sentences = []
        for paragraph in line:
            if args.unk is not None:
                paragraph = paragraph.replace(args.unk, "unk")
            sentences = sentences + sent_tokenize(paragraph.strip())

        sentences_to_return, sentences_split_to_return, count_overlong_internal, count_char_level_internal, count_word_tokenized_internal, count_space_delimited_internal, count_sentences_internal = sentences_trim(sentences)
        count_overlong += count_overlong_internal
        count_char_level += count_char_level_internal
        count_word_tokenized += count_word_tokenized_internal
        count_space_delimited += count_space_delimited_internal
        count_sentences += count_sentences_internal


        for sentence in sentences_to_return:
            fo.write(sentence + "\n")

        for sentence in sentences_split_to_return:
            fo_split.write(sentence + "\n")


print("Count of sentences that were too long:", count_overlong)
print("Count separated using spaces:", count_space_delimited)
print("Count separated using word tokenization:", count_word_tokenized)
print("Count separated at the character level:", count_char_level)
print("Count of sentences:", count_sentences)


