import argparse
  
parser = argparse.ArgumentParser()
parser.add_argument("--file", help="file to trim", type=str, default=None)
parser.add_argument("--linecount", help="number of lines to have in each file", type=int, default=None)
args = parser.parse_args()

fi = open(args.file, "r")


for line_number, line in enumerate(fi):
    if line_number % args.linecount == 0:
        if line_number != 0:
            fo.close()
        fo = open(args.file + "." + str(line_number//args.linecount + 1), "w")
        print(line_number//args.linecount + 1)
    fo.write(line)
                     
