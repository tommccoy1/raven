# RAVEN
RAting VErbal Novelty

This repo provides the code for our paper [How much do language models copy from their training data? Evaluating linguistic novelty in text generation using RAVEN](https://arxiv.org/pdf/2111.09509.pdf).


# Quickstart

0. Install dependencies (see next section)

1. We have provided some example generated text for which we want to evaluate the novelty: this is in `data/prompts_and_generations/example_prompts.txt` and `data/prompts_and_generations/example_generations.txt`. We will be evaluating its novelty relative to the mini training set in `data/miniwiki/wiki.train.tokens`. The first step is to annotate each word with its pointwise duplication score:
```
python pointwise_annotation_training.py --generation_file ../../data/prompts_and_generations/example_generations.txt --prompt_file ../../data/prompts_and_generations/example_prompts.txt --training_file ../../data/miniwiki/wiki.train.tokens --max_ngram_first_pass 10 --eos \<eos\>
python pointwise_annotation_context.py --generation_file ../../data/prompts_and_generations/example_generations.txt --prompt_file ../../data/prompts_and_generations/example_prompts.txt --to_score context
python pointwise_annotation_training_and_context.py --training_pointwise ../../data/prompts_and_generations/example_generations.txt.training_pointwise --context_pointwise ../../data/prompts_and_generations/example_generations.txt.context_pointwise
```

2. Now, using the annotated files above, generate a report describing the level of n-gram novelty. The resulting n-gram novelty report will be found in `../../data/prompts_and_generations/example_generations.txt.context_and_training_pointwise.ngram_report`.
```
python ngram_overlap_from_pointwise.py --pointwise_file ../../data/prompts_and_generations/example_generations.txt.context_and_training_pointwise --generation_file ../../data/prompts_and_generations/example_generations.txt --fast_analyses
```

3. In order to perform our syntactic novelty analyses, we need to parse our training set and our generated text. The parsers that we use take raw text (not tokenized), whereas our training set and generated text are tokenized. Therefore, we first need to detokenize the text (if you are using a different dataset that is not tokenized, this step might not be necessary):
```
python wikitext_detokenizer.py --file ../data/miniwiki/wiki.train.tokens
python wikitext_detokenizer.py --file ../data/prompts_and_generations/example_generations.txt
python wikitext_detokenizer.py --file ../data/prompts_and_generations/example_prompts.txt
```

4. Next we sentence tokenize and parse our training set and our generated text, giving both constituency and dependency parses. (On the mini files listed here, these two parsing commands will each take about 3 minutes).
```
python sentence_tokenize.py --filename ../data/miniwiki/wiki.train.tokens.detokenized --max_subword_count 400 --unk \<unk\>
python sentence_tokenize.py --filename ../data/prompts_and_generations/example_generations.txt.detokenized --max_subword_count 400 --generation --prompt_filename ../data/prompts_and_generations/example_prompts.txt.detokenized --newline \<eos\> --unk \<unk\> --space_before_generation

python parse.py --file_to_parse ../data/miniwiki/wiki.train.tokens.detokenized.sentences --training
python parse.py --file_to_parse ../data/prompts_and_generations/example_generations.txt.detokenized.sentences
```

5. Finally we use these parses to analyze the syntactic novelty of the generated text. The resulting syntactic novelty report will be found in `../../data/prompts_and_generations/example_generations.txt.detokenized.sentences.parsed.syntax_report`
```
python run_syntax_analyses.py --training ../../data/miniwiki/wiki.train.tokens.detokenized.sentences.parsed --generation ../../data/prompts_and_generations/example_generations.txt.detokenized.sentences.parsed --all_analyses
```


# Requirements/dependencies
1. Python 3 (we used Python 3.8.1)

2. Create a virtual environment in src/ and install requirements:
```
cd src

python -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel

# We used PyTorch version 1.8.1+cu102
pip install torch

# We used version 4.6.1
pip install transformers

pip install sentencepiece

# We used version 6.0.3
pip install ftfy

# We used version 3.6.2
pip install nltk

# We used version 1.1.1
# Only needed for analyzing syntactic novelty
pip install -U supar

# We used version 0.2.0
# Only needed for analyzing syntactic novelty
pip install benepar
python -m spacy download en_core_web_md

```

# Analyzing n-gram novelty for your own data

## Running the analyses

1. To analyze the n-gram novelty of your own data, you need to provide 3 files. All 3 of them should be tokenized using whatever tokens you want n-gram novelty to be computed over (e.g., if you want n-grams of subword tokens, then the files should contain subword tokens separated by spaces; if you want n-grams of words, then the files should be tokenized at the word level, with spaces separating words):
- The generated text: A text file with one passage of generated text per line (if these passages sometimes span multiple lines, you should replace the newlines within the generations with some special token, such as NEWLINE).
- (Optional) the prompts: A text file with one prompt per line, where the lines in this file match up with the lines in the file of generated text (if these prompts sometimes span multiple lines, you should replace the newlines within the prompts with some special token, such as NEWLINE). If you did not use prompts to generate your text, then you don't need this file.
- The training set: A text file.

2. The first step is to annotate your generated text with each token's pointwise duplication score. A token's pointwise duplication score is the size of the smallest n-gram ending in that token that is novel. For instance, if the token is *be* in *these rules will not ***be***,* and if the n-grams *be*, *not be*, and *will not be* all appeared in the training set, but *rules will not be* did not, then the pointwise duplication score for *be* would be 4.

We provide scripts to annotate novelty with respect to 3 types of sources: the training set (i.e., is each n-gram duplicated from the training set?); the context, which is the prompt plus all text that the model has previously generated from this prompt; and the union of the training set and the context (this final option is the main one considered in the paper, since we only wanted to count text as novel if it was not duplicated from either the training set or the context). These scripts will create 3 different annotated files: PATH_TO_GENERATION.training_pointwise, PATH_TO_GENERATION.context_pointwise, and PATH_TO_GENERATION.training_and_context_pointwise. Note that the 3rd command can only be run after the first 2 have been run.
```
python pointwise_annotation_training.py --generation_file PATH_TO_GENERATION --prompt_file PATH_TO_PROMPT --training_file PATH_TO_TRAINING --max_ngram_first_pass 10 --eos \<eos\>
python pointwise_annotation_context.py --generation_file PATH_TO_GENERATION --prompt_file PATH_TO_PROMPT --to_score context
python pointwise_annotation_training_and_context.py --training_pointwise PATH_TO_GENERATION.training_pointwise --context_pointwise PATH_TO_GENERATION.context_pointwise
```

Some important options for the `pointwise_annotation_training.py` script:
- `--max_ngram_first_pass VALUE`: The annotation procedure works in 2 passes: the first pass checks all n-grams up to a certain size, and then the second pass checks for n-grams larger than that. This parameter is the maximum used in the first pass. Its value should not affect the final result, but it might affect the speed. We recommed using a value of 10.
- `--first_pass_only`: The second pass of the annotation can take a very long time. Therefore, one way to speed it up is to only perform the first pass, which is achieved by including this option; in this case, you will only get results up to the n-gram size specified with `--max_ngram_first_pass`. For example, if `--max_ngram_first_pass` is 10, you will know what proportion of unigrams, bigrams, trigrams, 4-grams, ..., 10-grams are novel, but not for larger n-grams. If you go this route, then `pointwise_annotation_context.py` should also be given the parameter `--max_ngram VALUE` with whatever value you used for `--max_ngram_first_pass VALUE`, so that the context-based duplication score will also be truncated at the appropriate value.
- `--prompt_file PATH_TO_PROMPT`: This can be omitted if you did not use prompts to generate your text.
- `--eos EOS_TOKEN`: If you have a specific EOS token that is in your generated text but not in your training file, specify it with this argument; otherwise, omit this argument. In our Wikitext examples, there is such a token: <eos> (which, when passed in at the command line, needs to have its brackets escaped as \<eos\>).

3. Once you have your annotated files, you can now run the n-gram novelty analyses. This will generate a report at `PATH_TO_POINTWISE_FILE.ngram_report`:
```
python ngram_overlap_from_pointwise.py --pointwise_file PATH_TO_POINTWISE_FILE --generation_file PATH_TO_GENERATION --fast_analyses
```

You can use command-line arguments to select which analyses are performed. For details on the options, run `python ngram_overlap_from_pointwise.py -h`. 

For PATH_TO_POINTWISE_FILE, you should select whichever type of pointwise file you want novelty analyses from (`.training_pointwise`, `.context_pointwise`, or `.training_and_context_pointwise`). 

## Handling enormous data

If your dataset is enormous, running the full suite of analyses might take a very long time. Here is what we recommend if you want to get some results that will be not as complete but that can also be run more quickly:
- When running `pointwise_annotation_training.py`, use the options `--max_ngram_first_pass 10` and `--first_pass_only`. This will restrict the results to give only the proportion of n-grams that are novel for n <= 10; but we found that these values of n are enough to give a reasonably clear sense of the model's novelty.
- When running `ngram_overlap_from_pointwise.py`, use just the option `--ngram_overlap`.

Using these options will restrict the results to give only the proportion of n-grams that are novel for n <= 10; but we found that these values of n are enough to give a reasonably clear sense of the model's novelty.


## Interpreting the results

Here is a list of all the results that might be listed in the `.ngram_report` file, depending on what options you passed to `ngram_overlap_from_pointwise.py`. Note that all proportions are proportions of tokens, not types (i.e., occurrences, not unique occurrences):
- Perplexity: The perplexity of the generated text
- Pointwise score: The average pointwise duplication score of all tokens. A token's pointwise duplication score is the size of the smallest n-gram ending in that token that is novel. For instance, if the token is *be* in *these rules will not ***be***,* and if the n-grams *be*, *not be*, and *will not be* all appeared in the training set, but *rules will not be* did not, then the pointwise duplication score for *be* would be 4.
- Inverse pointwise score: The average pointwise duplication score of all tokens, except raising each score to the power -1 before averaging (this is one approach to handle large outliers).
- Log2 pointwise score: The average pointwise duplication score of all tokens, except first taking the base-2 log of each score before averaging (this is one approach to handle large outliers).
- Ln pointwise score: The average pointwise duplication score of all tokens, except first taking the natural log of each score before averaging (this is one approach to handle large outliers).
- Truncated-5 pointwise score: The average pointwise duplication score of all tokens, except first truncating each score at 5 before averaging (this is one approach to handle large outliers, and is the one we used in the paper). 
- Average pointwise score by position: For each position in generated text, lists the mean pointwise duplication score for tokens in that position (e.g., the mean score for tokens in the first position, averaged across all generations).
- Binned average pointwise score by position: Groups the positions into bins of 100, and then gives the average pointwise score for each bin.
- Truncated average pointwise score by position: For each position in generated text, lists the mean pointwise duplication score for tokens in that position, but truncating the scores at 10 before averaging
- Truncated binned average pointwise score by position: Same as ``binned average pointwise score by position" except truncating the scores at 10 before averaging
- Overlap sizes: Lists the sizes of all n-grams (this is paired with the line below it, ``Overlap proportions")
- Overlap proportions: For each n-gram size, gives the proportion of n-grams of that size that are duplicated. The first number gives the proportion of unigrams, the second number gives the proportion of bigrams, etc.
- Novel bigram examples in context: Gives examples of novel bigrams in context (where the bigrams are surrounded by asterisks).
- Supercopying examples in context: Gives exampels of supercopying in context (where the supercopied text is surrounded by asterisks).
- Length of longest supercopying example: Gives the length in tokens of the largest supercopied passage across all the generations
- Longest supercopying example: Gives the text of the longest supercopied passage
- Average supercopying overlap: Across all supercopied 100-grams, gives the average number of times that the 100-gram appears in the training set
- Average supercopying max overlap: For each supercopied passage, find the 100-gram within that passage that appears the most times in the training set and return the number of times it appears in the training set; this number gives the average of these maximum values across all supercopied passages
- Average random overlap: Across 1,000 randomly-selected 100-grams from the training set, gives the average number of times that each appears in the training set (so this is guaranteed to be at least 1, since each must appear in the training set at least once; a number larger than one means that the passage is repeated in the training set).
- All supercopying overlaps: For each supercopied 100-gram, lists how many times it appears in the training set
- All supercopying max overlaps: For each supercopied passage, lists the number of appearances in the training set had by the 100-gram within the supercopied passage that appears the most times in the training set
- All random overlaps: For each of 1,000 randomly-selected 100-grams in the training set, gives how many times that 100-gram appears in the training set.
- Max supercopy overlap: Gives the maximum number of times that a supercopied 100-gram appears in the training set.
- Max supercopy: Gives the text of the supercopied passage that appears the most times in the training set.

# Analyzing syntactic novelty for your own data

## Running the analyses
1. To analyze the syntactic novelty of your own data, you need to provide 3 files. All 3 of them should be raw text - that is, not tokenized (the parsers that we use do their own tokenizing, which will not work as well if the text is already tokenized). If your model generates text in tokenized way, you might need to detokenize that text.
- The generated text: A text file with one passage of generated text per line (if these passages sometimes span multiple lines, you should replace the newlines within the generations with some special token, such as NEWLINE).
- (Optional) the prompts: A text file with one prompt per line, where the lines in this file match up with the lines in the file of generated text (if these prompts sometimes span multiple lines, you should replace the newlines within the prompts with some special token, such as NEWLINE). If you did not use prompts to generate your text, then you don't need this file.
- The training set: A text file.

2. The first step is to sentence-tokenize the training set and generated text:
```
python sentence_tokenize.py --filename PATH_TO_TRAINING --max_subword_count 400 --unk \<unk\>
python sentence_tokenize.py --filename PATH_TO_GENERATION --max_subword_count 400 --generation --prompt_filename PATH_TO_PROMPT --newline \<eos\> --unk \<unk\> --space_before_generation
```

Here is a description of the options for running `sentence_tokenize.py`:
- `--max_subword_count 400`: This gives the maximum number of T5 subwords per sentence; longer sentences will be broken into several sentences. This is necessary because the parsers cannot handle sentences that are too long.
- `--prompt_filename PATH_TO_PROMPT`: You can omit this if you did not generate from prompts.
- `--generation`: Include if the file being processed is a file of generated text (not a training set)
- `--space_before_generation`: Whether a space should be added between the prompt and the generation (that is, there should be a space there, but it is not already present at the start of each generated line)
- `--newline`: If your prompts and generations have a special token that means a newline, specify it here. E.g., Wikitext uses <eos> (which, on the command line, is specified as \<eos\>).
- `--unk`: If your text uses a special unk token, specify it here. E.g., Wikitext uses <unk> (which, on the command line, is specified as \<unk\>).

3. Next we need to parse the training set and generated text. Include the option `--training` if the file being parsed is a training-set file.
```
python parse.py --file_to_parse PATH_TO_TRAINING.sentences --training
python parse.py --file_to_parse PATH_TO_GENERATION.sentences
```

4. Finally, run the syntax analyses. For now, the only option is to run `--all_analyses` (the code would require some refactoring to have it run only certain sub-analyses). This will generate a report at `PATH_TO_POINTWISE_FILE.syntax_report`:
```
python run_syntax_analyses.py --training PATH_TO_TRAINING.sentences.parsed --generation PATH_TO_GENERATION.sentences.parsed --all_analyses
```

## Handling enormous datasets

For syntactic novelty, by far the most time-consuming part is parsing the training set. If your training set is large, we recommend first parsing just a small portion of it to see how long it takes, and then using that to estimate how long the entire training set will take to parse to judge if it will be reasonable. If you have access to multiple CPUs or GPUs, you could split your training set into multiple parts and then parse one part per machine to speed up parsing (each sentence is parsed independently, so there is no harm in parallelizing the parsing in this way). You can then combine all of the parsed training sentences back into one file to run the syntax analyses.

## Interpreting the results

Here is a list of all the results that will be listed in the `.syntax_report` file. Note that all proportions are proportions of tokens, not types (i.e., occurrences, not unique occurrences). Be warned that some of these numbers might be unreliable due to parsing errors - see our paper (Section 6) for details:
- Novel POS tags: Gives the proportion of words in the generation that appear with a POS tag they never had during training. (This excludes generated words which never appeared in training at all).
- Examples of novel POS tags: Gives examples of words with novel POS tags.
- Examples of novel nouns: Gives examples of words that were generated with any noun tag (NN, NNS, NNP, NNPS) when it never had one of those tags during training.
- Examples of novel verbs: Gives examples of words that were generated with any verb tag (VB, VBN, VBZ, VBD) when it never had one of those tags during training.
- Novel CFG rules: Gives the proportion of CFG rules used in the constituency parses of the generated text that never appeared in the training set. Only rules that consist entirely of nonterminals are considered (i.e., we exclude the rules used to generate words at the leaves of the trees).
- Examples of novel CFG rules: Gives examples of novel CFG rules.
- Examples of novel unary and binary CFG rules: Gives examples of novel CFG rules whose right hand sides have only 1 or 2 nonterminals.
- Novel POS sequences: Gives the proportion of generated sentences for which the sequence of part-of-speech tags in that sentence was novel (i.e., there was no training sentence with that sequence of tags).
- Examples of novel POS sequences: Gives examples of generated sentences with novel POS sequences.
- Novel constituency structures: Gives the proportion of generated sentences for which the constituency structure (i.e., the parse tree minus the words at the leaves) was novel.
- Examples of novel constituency structures: Gives examples of generated sentences with novel constituency structures.
- Novel dependency arcs (labeled): Gives the proportion of labeled dependency arcs in the generated sentences that are novel. A labeled dependency arc is a 3-tuple containing the two words at the ends of the arc as well as the arc's label, such as (the, dog, det).
- Examples of novel dependency arcs: Gives examples of novel labeled dependency arcs
- Examples of novel dependency arcs - a to the: Gives examples where a noun that appeared with the determiner *a* or *an* in training but never with *the* now appears with *the*
- Examples of novel dependency arcs - the to a: Gives examples where a noun that appeared with the determiner *the* in training but never with *a* or *an* now appears with *a* or *an*
- Examples of novel dependency arcs - novel subject: Gives examples of a verb appearing with a word as its subject that was never its subject during training
- Examples of novel dependency arcs - novel object: Gives examples of a verb appearing with a word as its object that was never its object during training
- Novel dependency arcs (unlabeled): Gives the proportion of unlabeled dependency arcs in the generated sentences that are novel. An unlabeled dependency arc is a 2-tuple containing the two words at the ends of the arc, such as (the, dog).
- Examples of novel dependency arcs (unlabeled): Gives examples of novel unlabeled dependency arcs.
- Novel dependency roles: Gives the proportion of word/[arc label and position in arc] 2-tuples that are novel; each labeled dependency arc gives 2 such 2-tuples, one for each word on its ends. An example of such a 2-tuple would be `dog as the dependent in a det dependency`
- Examples of novel dependency roles: Gives examples of novel word/[arc label and position in arc] 2-tuples. 
- Examples of novel dependency relations - nsubj to obj: Gives examples of a word being used as a direct object when it never was one during training (but was a subject).
- Examples of novel dependency relations - obj to nsubj: Gives examples of a word being used as a subject when it never was one during training (but was a direct object).
- Examples of novel dependency relations - active to passive: Gives examples of a verb being used in the passive voice when it was never passive during training (but was used in the active voice)
- Examples of novel dependency relations - passive to active: Gives examples of a verb being used in the active voice when it was never active during training (but was used in the passive voice)
- Novel dependency argument structures: Gives the proportion of generated verbs appearing with a novel argument structure, where the argument structure is the list of arguments it has (e.g., ``subject and direct object")
- Examples of novel argument structures: Gives examples of generated verbs with novel argument structures
- Examples of novel argument structures - transitive to intransitive: Gives examples of verbs being used intransitively when they never were used this way during training (but were used transitively)
- Examples of novel argument structures - intransitive to transitive: Gives examples of verbs being used transitively when they never were used this way during training (but were used intransitively)
- Examples of novel argument structures - do to po: Gives examples of verbs being used with a [prepositional object + direct object] (po) when they never had this argument structure during training, but did appear in training with a double object construction (do)
- Examples of novel argument structures - po to do: Gives examples of verbs being used with a double object construction (do) when they never had this argument structure during training, but did appear in training with a [prepositional object + direct object] (po) argument structure 








# Replicating our Wikitext-103 experiments

This section goes over how to re-run our experiments with models trained on Wikitext-103 (i.e., the LSTM, Transformer, and Transformer-XL experiments reported in our paper). Note that some of these commands will take a very long time to run. In particular, we have grouped together related commands into .sh files, but in many cases it would take several weeks to run the entire .sh file, so you may instead wish to run the commands inside each file individually (none of the commands within an .sh file depend on each other, so they can be run in parellel).

1. Download Wikitext-103 (in data/):
```
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
rm wikitext-103-v1.zip
```

2. Create prompts
```
python prompt_creation.py --nprompts_per_file 1000 --nfiles 1 --prompt_length 0 --continuation_length 1010 --source_file ../data/wikitext-103/wiki.test.tokens --token_before_prompt \<eos\> --newline_token \<eos\> --filetype wikitext --output_prefix ../data/prompts_and_generations/wikitext
python prompt_creation.py --nprompts_per_file 1000 --nfiles 1 --prompt_length 16 --continuation_length 1010 --source_file ../data/wikitext-103/wiki.test.tokens --token_before_prompt \<eos\> --newline_token \<eos\> --filetype wikitext --output_prefix ../data/prompts_and_generations/wikitext
python prompt_creation.py --nprompts_per_file 1000 --nfiles 1 --prompt_length 128 --continuation_length 1010 --source_file ../data/wikitext-103/wiki.test.tokens --token_before_prompt \<eos\> --newline_token \<eos\> --filetype wikitext --output_prefix ../data/prompts_and_generations/wikitext
python prompt_creation.py --nprompts_per_file 1000 --nfiles 1 --prompt_length 512 --continuation_length 1010 --source_file ../data/wikitext-103/wiki.test.tokens --token_before_prompt \<eos\> --newline_token \<eos\> --filetype wikitext --output_prefix ../data/prompts_and_generations/wikitext
```

3. Generate text
```
# Transformer-XL
source generate_transfo-xl.sh

# Transformer
python add_eos.py --file ../data/wikitext-103/wiki.train.tokens
python add_eos.py --file ../data/wikitext-103/wiki.valid.tokens
python add_eos.py --file ../data/wikitext-103/wiki.test.tokens
mv ../data/wikitext-103/wiki.train.tokens.eos ../data/wikitext-103/train.txt
mv ../data/wikitext-103/wiki.valid.tokens.eos ../data/wikitext-103/valid.txt
mv ../data/wikitext-103/wiki.test.tokens.eos ../data/wikitext-103/test.txt
cp ../data/wikitext/vocab.txt ../data/wikitext-103/

source generate_transformer.sh

# LSTM: Note that this uses the LSTM that we trained, which we have not yet released, but will update here when we have released it
git clone https://github.com/facebookresearch/colorlessgreenRNNs.git
cp generate_lstm.py colorlessgreenRNNs/src/language_models/
cp lstm_lm.pt colorlessgreenRNNs/src/language_models/
cp generate_lstm.sh colorlessgreenRNNs/src/language_models/
cd colorlessgreenRNNs/src/language_models/

python -m venv .venv
source .venv/bin/activate
pip install torch==1.4.0 torchvision==0.5.0
source generate_lstm.sh
```

4. Trim continuations and generations
```
source trim_wikitext_continuations.sh

source txl_trim.sh
source lstm_trim.sh
source transformer_trim.sh
```

5. Pointwise annotate
```
# Training set
source lstm_training_pointwise.sh
source transformer_training_pointwise.sh
source txl_training_pointwise.sh
source wikitext_continuations_training_pointwise.sh

# Context
source lstm_context_pointwise.sh 
source transformer_context_pointwise.sh
source txl_context_pointwise.sh
source wikitext_continuations_context_pointwise.sh

# Training and context
source lstm_training_and_context_pointwise.sh
source transformer_training_and_context_pointwise.sh
source txl_training_and_context_pointwise.sh
source wikitext_continuations_training_and_context_pointwise.sh
```

6. N-gram analyses
```
# Training set
source lstm_ngram_analysis_training.sh
source transformer_ngram_analysis_training.sh
source txl_ngram_analysis_training.sh
source wikitext_continuations_ngram_analysis_training.sh

# Context
source lstm_ngram_analysis_context.sh
source transformer_ngram_analysis_context.sh
source txl_ngram_analysis_context.sh
source wikitext_continuations_ngram_analysis_context.sh

# Training and context
source lstm_ngram_analysis_context_and_training.sh
source transformer_ngram_analysis_context_and_training.sh
source txl_ngram_analysis_context_and_training.sh
source wikitext_continuations_ngram_analysis_context_and_training.sh

```


7. Detokenize
```
python wikitext_detokenizer.py --file ../data/wikitext-103/wiki.valid.tokens
python wikitext_detokenizer.py --file ../data/wikitext-103/wiki.test.tokens
python wikitext_detokenizer.py --file ../data/wikitext-103/wiki.train.tokens

python wikitext_detokenizer.py --file ../data/prompts_and_generations/wikitext_prompts_length0_1of1.txt --newline \<eos\>
python wikitext_detokenizer.py --file ../data/prompts_and_generations/wikitext_prompts_length16_1of1.txt --newline \<eos\>
python wikitext_detokenizer.py --file ../data/prompts_and_generations/wikitext_prompts_length128_1of1.txt --newline \<eos\>
python wikitext_detokenizer.py --file ../data/prompts_and_generations/wikitext_prompts_length512_1of1.txt --newline \<eos\>

python wikitext_detokenizer.py --file ../data/prompts_and_generations/wikitext_continuations_length0_1of1.txt.trimmed --newline \<eos\>
python wikitext_detokenizer.py --file ../data/prompts_and_generations/wikitext_continuations_length16_1of1.txt.trimmed --newline \<eos\>
python wikitext_detokenizer.py --file ../data/prompts_and_generations/wikitext_continuations_length128_1of1.txt.trimmed --newline \<eos\>
python wikitext_detokenizer.py --file ../data/prompts_and_generations/wikitext_continuations_length512_1of1.txt.trimmed --newline \<eos\>

source lstm_detokenize.sh
source txl_detokenize.sh
source transformer_detokenize.sh

python divide_by_linecount.py --file ../data/wikitext-103/wiki.train.tokens.detokenized --linecount 20000
```

8. Sentence tokenize and parse
```
source wikitext_training_sent_tokenize.sh
source wikitext_continuations_sentence_tokenize.sh
source lstm_sentence_tokenize.sh
source transformer_sentence_tokenize.sh
source txl_sentence_tokenize.sh

source wikitext_training_parse.sh
python combine_wikitext_parse.py

source wikitext_continuations_parse.sh
source lstm_parse.sh
source transformer_parse.sh
source txl_parse.sh

```

9. Syntax analyses
```
source lstm_syntax.sh
source transformer_syntax.sh
source txl_syntax.sh
source wikitext_continuations_syntax.sh
```


# GPT-2 experiments

Because GPT-2 was trained on data that is not publicly released, we are unable to provide the data used as inputs to our analyses, but we still provide the commands used to generate text from GPT-2 and to analyze that text.

4. Generate text
```
source generate_gpt2.sh
```

5. Trim continuations and generations
```
source trim_gpt2.sh
source trim_webtext_continuations.sh
```

6. Convert BPE indices to words
```
source all_gen_detokenization_redo.sh
```


10. Tokenize
```
# Moses tokenizing GPT-2 generations
source gpt2_gen_moses.sh

```

11. Pointwise annotate
```
# Training set
source gpt2_pointwise_commands.sh
source webtext_continuations_pointwise_commands.sh

# Context
source gpt2_pointwise_commands_context.sh
source webtext_continuations_pointwise_commands_context.sh

# Training and context
source gpt2_pointwise_commands_training_and_context.sh
source webtext_continuations_pointwise_commands_training_and_context.sh

```

12. N-gram analyses
```
# Training set
source gpt2_ngram_analysis_training.sh
source webtext_continuations_ngram_analysis_training.sh

# Context
source gpt2_ngram_analysis_context.sh
source webtext_continuations_ngram_analysis_context.sh

# Training and context
source gpt2_ngram_analysis_context_and_training.sh
source webtext_continuations_ngram_analysis_context_and_training.sh
```



13. Sentence tokenize
```
gpt2_gen_sent_tokenize.sh
```

14. Parse
```
gpt2_gen_parse.sh
```


15. Run syntax analyses
```
gpt2_syntax_commands.sh
```



## License

This code is licensed under an [MIT license](https://github.com/tommccoy1/raven/blob/main/LICENSE).

## Citing

If you make use of this code, please cite the following ([bibtex](https://tommccoy1.github.io/raven_bib.html)):

R. Thomas McCoy, Paul Smolensky, Tal Linzen, Jianfeng Gao, and Asli Celikyilmaz.  2021. How much do language models copy from their training data? Evaluating linguistic novelty in text generation using RAVEN  *arXiv preprint arXiv 2111.09509*.

*Questions? Comments? Email [tom.mccoy@jhu.edu](mailto:tom.mccoy@jhu.edu).*















