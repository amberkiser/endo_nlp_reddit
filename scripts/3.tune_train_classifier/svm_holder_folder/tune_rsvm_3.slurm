#!/bin/tcsh
#SBATCH --account=bmi_abdelrahman
#SBATCH --partition=kingspeak
#SBATCH --output=slurmjob-%j
#SBATCH --job-name=rsvm3

setenv WORKDIR $HOME/EndoReddit/scripts/tune_train_classifier

module use $HOME/MyModules
module load miniconda3/latest

srun python tune_rsvm_3.py