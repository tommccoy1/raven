
fo = open("../data/webtext/webtext_train_tokens.npy.tokens.word", "w")

for i in range(19):
    num_string = str(i)
    print(num_string)

    if len(num_string) == 1:
        num_string = "0" + num_string

    fi = open("../data/webtext/webtext_" + num_string + "_tokens.npy.tokens.word", "r")

    for line in fi:
        fo.write(line)


