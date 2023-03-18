
training = "../../data/webtext/webtext_train_tokens.npy.tokens.word.parsed.combined"
lengths = ["length0", "length18", "length141", "length564"]

fo = open("webtext_syntax.sh", "w")

for length in lengths:
    generation = "webtext_continuations_" + length + "_1of1.txt.trimmed.word.sentences.parsed"
    generation = "../../data/prompts_and_generations/" + generation

    command = "python run_syntax_analyses.py --training " + training + " --generation " + generation + " --all_analyses"
    command_name = "_".join(["webtext", "syntax", length]) 
    fo.write(command + "\t" + command_name + "\n")




