#!/bin/bash -l

#$ -P cs505Gantry   # Specify the SCC project name you want to use
#$ -l h_rt=48:00:00   # Specify the hard time limit for the job
#$ -N ian_isaac           # Give job a name
#$ -pe omp 8              # Request 8 CPU core
#$ -l gpu_card=1         # Request 1 GPU

module load miniconda
module load academic-ml
conda activate spring-2025-pyt

python -u train_test_transformer.py