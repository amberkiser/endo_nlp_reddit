def makefilio(fileNum):

    line01 = "#!/bin/tcsh"
    line02 = "#SBATCH --account=bmi_abdelrahman"
    line03 = "#SBATCH --partition=kingspeak"
    line04 = "#SBATCH --output=slurmjob-%j"
    line05 = "#SBATCH --job-name=nn%d" %fileNum
    line06 = ""
    line07 = "setenv WORKDIR $HOME/EndoReddit/scripts/tune_train_classifier"
    line08 = ""
    line09 = "module use $HOME/MyModules"
    line10 = "module load miniconda3/latest"
    line11 = ""
    line12 = "srun python tune_nn_%d.py" %fileNum

    lines = [line01, line02, line03, line04, line05, line06, line07, line08, line09, line10, line11, line12]

    with open("tune_nn_%d.slurm" % fileNum, "w") as out:
        out.write('\n'.join(lines))


for i in range(1, 46, 1):
    makefilio(i)
