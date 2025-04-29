#!/bin/bash -l

#$ -P cs505aw       # Specify the SCC project name you want to use
#$ -l h_rt=48:00:00   # Specify the hard time limit for the job
#$ -N ian_isaac           # Give job a name
 
module load miniconda
module load academic-ml
conda activate spring-2025-pyt

python train_test_transformer.py