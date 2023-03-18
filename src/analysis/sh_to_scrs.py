


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--command_file", help="file of commands to turn into MARCC scripts", type=str, default=None)
parser.add_argument("--partition", help="partition to run commands in", type=str, default="defq")
parser.add_argument("--repetitions", help="number of times to repeat each command", type=int, default=1)
parser.add_argument("--tasks_per_node", help="tasks per node", type=int, default=1)
parser.add_argument("--cd_up", action="store_true", help="Whether to cd to the parent directory for running the command")
parser.add_argument("--titles", action="store_true", help="Whether the file contains titles for the files")
parser.add_argument("--hours", help="hours for the job to run", type=str, default="72")
args = parser.parse_args()

fi = open(args.command_file, "r")
prefix = args.command_file[:-3]

to_run = open(prefix + "_run_all.sh", "w")

for index, line in enumerate(fi):
    if args.titles:
        parts = line.strip().split("\t")
        this_name = parts[1]
        command = parts[0]
    else:
        command = line.strip()
        this_name = None

    if command == "":
        continue

    else:
        if this_name is None:
            this_name = prefix + "_" + str(index+1)
        fo = open(this_name + ".scr", "w")
        to_run.write("sbatch " + this_name + ".scr\n")

        fo.write("#!/bin/bash\n")
        fo.write("#SBATCH --job-name=" + this_name + "\n")
        fo.write("#SBATCH --partition=" + args.partition + "\n")
        fo.write("#SBATCH --time=" + args.hours + ":0:0\n")
        
        if "gpu" in args.partition:
            fo.write("#SBATCH --gres=gpu:1\n")
        else:
            fo.write("#SBATCH --nodes=1\n")

        fo.write("#SBATCH --ntasks-per-node=" + str(args.tasks_per_node) + "\n")
        fo.write("#SBATCH --mail-type=end\n")
        fo.write("#SBATCH --mail-user=rmccoy20@jhu.edu\n")
        fo.write("#SBATCH --output=" + this_name + ".log\n")
        fo.write("#SBATCH --error=" + this_name + ".err\n\n")
        fo.write("module load python\nsource ../.venv/bin/activate\n")
        

        if args.cd_up:
            fo.write("cd ..\n")

        for _ in range(args.repetitions):
            fo.write(command + "\n")

