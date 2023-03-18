

fo = open("gpt2_ngram_analysis_training.sh", "w")
for model, batchsize in [("gpt2", 50), ("gpt2-medium", 32), ("gpt2-large", 16), ("gpt2-xl", 5)]:
    for prompt_length in ["0", "18", "141", "564"]:
        for gen_type in ["k1_p1_temp1.0", "k10_p1_temp1.0", "k40_p1_temp1.0", "k800_p1_temp1.0", "k1234567890_p0.75_temp1.0", "k1234567890_p0.9_temp1.0", "k1234567890_p0.95_temp1.0", "k1234567890_p1.0_temp1.0", "k1234567890_p1.0_temp0.7", "k1234567890_p1.0_temp0.9", "k1234567890_p1.0_temp1.1", "k1234567890_p1.0_temp1.3"]:
            genfile = "../../data/prompts_and_generations/" + model + "_webtext_prompts_length" + prompt_length + "_1of1_" + gen_type + "_beam1_len1300_batchsize" + str(batchsize) + ".generated.trimmed.word"
            pointwisefile = genfile + ".moses.pointwise"
            promptfile = "../../data/prompts_and_generations/webtext_prompts_length" + prompt_length + "_1of1.txt.word"
            trainfile = "../../data/webtext/webtext_train_tokens.npy.tokens.word.moses.combined"
            perplexity_model = "transfo-xl-wt103"
            eos_token = "\\&NEWLINE\\;"

            command = "python ngram_overlap_from_pointwise.py --pointwise_file " + pointwisefile + " --generation_file " + genfile + " --prompt_file " + promptfile + " --training_file " + trainfile + " --perplexity_model " + perplexity_model + " --eos_token " + eos_token + " --fast_analyses --supercopying_overlap"

            fo.write(command + "\n")




