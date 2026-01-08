#!/bin/bash
#SBATCH -J fit
#SBATCH -D ./
#SBATCH -o ./logs/job.out%A_%a
#SBATCH -e ./logs/job.out%A_%a
#SBATCH --partition=general
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=256
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kuhlmann@mpp.mpg.de

module purge
module load gcc/13
module load anaconda/3/2021.11
module load texlive/2021
module load parallel/201807
source /u/jdk/hnu/bin/activate
export TMPDIR=/ptmp/jdk

# python run.py --idx $IDX --setup
file="$(sed "${SLURM_ARRAY_TASK_ID}q;d" prior_fits.txt)"
echo $file
# parallel --delay 20 --link srun --exclusive -N 1 -n 1 python run.py $file --random {} ::: 1337 4242 ::: 6969 6666
# parallel --delay 20 --link srun --exclusive -N 1 -n 1 python run.py $file --random 1337 4242 6969 6666 -mufac {1} -sigma {2} ::: 1 1 10 10 ::: 4 6 4 6
srun --exclusive -N 1 -n 1 python run.py $file --random 1337 4242 6969 6666

