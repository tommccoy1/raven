
fo = open("../data/webtext/webtext_train_tokens.npy.tokens.word.moses.combined", "w")

def remove_excess_spaces(line):
    if "  " in line:
        return remove_excess_spaces(line.replace("  ", " "))
    else:
        return line


for i in range(496):
    print(i+1)
    fi = open("../data/webtext/webtext_train_tokens.npy.tokens.word." + str(i+1) + ".moses", "r")

    for line in fi:
        fo.write(line)


