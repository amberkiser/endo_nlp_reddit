#!/bin/tcsh
#SBATCH --account=bmi_abdelrahman
#SBATCH --partition=lonepeak
#SBATCH --output=slurmjob-%j
#SBATCH --job-name=rf17

setenv WORKDIR $HOME/EndoReddit/scripts/tune_train_classifier

module use $HOME/MyModules
module load miniconda3/latest

srun python tune_rf_17.py