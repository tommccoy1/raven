
We used the miniature dataset in this directory to check our analysis scripts; those scripts use some tricks to make the computation more memory-efficient, so here we compare the outputs of these scripts to a brute-force method (using a dataset small enough for the brute-force method to not take up too much memory).

The commands below should be run in the parent directory. Both diff commands should return no differences. The `gold` files are the results of running `pointwise_annotation_exhaustive.py`.

```
python pointwise_annotation.py --generation_file script_testing/generation_numbers.txt --prompt_file script_testing/prompt_numbers.txt --training_file script_testing/training_numbers.txt --max_ngram 10
diff script_testing/generation_numbers.txt.pointwise script_testing/gold_generation_numbers.txt.pointwise

python pointwise_annotation.py --generation_file script_testing/generation_numbers.txt --prompt_file script_testing/prompt_numbers.txt --training_file script_testing/training_numbers.txt --max_ngram 10 --eos \<eos\>
diff script_testing/generation_numbers.txt.pointwise script_testing/gold_generation_numbers.txt.pointwise_eos


```



