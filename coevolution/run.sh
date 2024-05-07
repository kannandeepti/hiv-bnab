#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -o logs/%j.out


source activate biopython

python -u run.py $1 $2 >& logs/${1}_${2}.out
