
training = "../../data/webtext/webtext_train_tokens.npy.tokens.word.parsed.combined"
sizes = [("gpt2", 50), ("gpt2-medium", 32), ("gpt2-large", 16), ("gpt2-xl", 5)]
lengths = ["length0", "length18", "length141", "length564"]
methods = ["k1_p1_temp1.0", "k10_p1_temp1.0", "k40_p1_temp1.0", "k800_p1_temp1.0",
           "k1234567890_p0.75_temp1.0", "k1234567890_p0.9_temp1.0", "k1234567890_p0.95_temp1.0", "k1234567890_p1.0_temp1.0",
           "k1234567890_p1.0_temp0.7", "k1234567890_p1.0_temp0.9", "k1234567890_p1.0_temp1.1", "k1234567890_p1.0_temp1.3"]

fo = open("gpt2_syntax.sh", "w")

for size, batch_size in sizes:
    for length in lengths:
        for method in methods:
            generation = "_".join([size, "webtext_prompts", length, "1of1", method, "beam1_len1300_batchsize" + str(batch_size)])
            generation = "../../data/prompts_and_generations/" + generation + ".generated.trimmed.word.sentences.parsed"

            command = "python run_syntax_analyses.py --training " + training + " --generation " + generation + " --all_analyses"
            command_name = "_".join([size, "syntax", length, method]) 
            fo.write(command + "\t" + command_name + "\n")




