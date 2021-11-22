
fo = open("../data/wikitext-103/wiki.train.tokens.detokenized.sentences.parsed.combined", "w")

for i in range(91):
    print(i+1)
    fi = open("../data/wikitext-103/wiki.train.tokens.detokenized." + str(i+1) + ".sentences.parsed", "r")

    for line in fi:
        fo.write(line)


