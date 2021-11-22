# RAVEN
RAting VErbal Novelty


# DELETE
sh_to_scrs.py
src/.venv
src/colorlessGreenRNNs
Un-trim GPT-2 pointwise commands?

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
pip install -U supar

# We used version 0.2.0
pip install benepar
python -m spacy download en_core_web_md

```

# Replicating our Wikitext-103 experiments

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

# LSTM
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







5. Detokenize
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

6. Sentence tokenize and parse
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

7. Syntax analyses
```
source lstm_syntax.sh
source transformer_syntax.sh
source txl_syntax.sh
source wikitext_continuations_syntax.sh
```


# Replicating GPT-2 experiments
1. Download numpy files
```
source azure_env.sh
python download_blob_conn_string.py
```

2. Convert numpy to text
```
source webtext_npy_to_txt.sh
```

3. Create prompts
```
python prompt_creation.py --nprompts_per_file 1000 --nfiles 1 --prompt_length 0 --continuation_length 1300 --source_file ../data/webtext/webtext_19.npy --token_before_prompt 50256 --filetype webtext --output_prefix ../data/prompts_and_generations/webtext
python prompt_creation.py --nprompts_per_file 1000 --nfiles 1 --prompt_length 18 --continuation_length 1300 --source_file ../data/webtext/webtext_19.npy --token_before_prompt 50256 --filetype webtext --output_prefix ../data/prompts_and_generations/webtext
python prompt_creation.py --nprompts_per_file 1000 --nfiles 1 --prompt_length 141 --continuation_length 1300 --source_file ../data/webtext/webtext_19.npy --token_before_prompt 50256 --filetype webtext --output_prefix ../data/prompts_and_generations/webtext
python prompt_creation.py --nprompts_per_file 1000 --nfiles 1 --prompt_length 564 --continuation_length 1300 --source_file ../data/webtext/webtext_19.npy --token_before_prompt 50256 --filetype webtext --output_prefix ../data/prompts_and_generations/webtext
```

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
source webtext_training_bpe_to_words.sh
source all_gen_detokenization_redo.sh
```

7. Combine them all into a single file
```
python combine_webtext.py
```

8. Divide training files into small ones that can be Moses-tokenized.
```
python divide_by_linecount.py --file ../data/webtext/webtext_train_tokens.npy.tokens.word --linecount 687000
```

10. Tokenize
```
# Moses tokenizing Webtext and GPT-2 generations
source webtext_training_moses.sh
python combine_webtext_moses.py

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









11. Divide training files into even smaller ones that can be sentence-tokenized and parsed.
```
python divide_by_linecount.py --file ../data/webtext/webtext_train_tokens.npy.tokens.word --linecount 100000
```



13. Sentence tokenize
```
gpt2_gen_sent_tokenize.sh
```




