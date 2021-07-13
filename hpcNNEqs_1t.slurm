#!/bin/bash
#SBATCH -J nnEq # Name for your job
#SBATCH -n 1 # Number of tasks when using MPI. Default is 1, and SLURM assumes the usage of 1 cpu per task.
#SBATCH -N 1 # Number of nodes to spread cores across - default is 1 - if you are not using MPI this should likely be 1
#SBATCH --mem 6000 # Megabytes of memory requested. Default is 2000/task.
#SBATCH -t 1-0:00:00 # Runtime in days-hours:minutes:seconds.
#SBATCH -p defq  # Partition to submit to the standard compute node partition(defq) or the express node partition(express)
#SBATCH --mail-user alexander.salois@student.montana.edu # this is the email you wish to be notified.
#SBATCH --mail-type FAIL,END # this specifies what events you should get an email about ALL will alert you of job beginning, completion, failure, etc.
#SBATCH -o nnEq1t_%A_%a.out.txt # Standard output
#SBATCH -e nnEq1t_%A_%a.err.txt # Standard error

date
echo "Hello from $(hostname)."
echo "${SLURM_ARRAY_TASK_ID}"

module load Anaconda3
source activate $HOME/condatf2
export HDF5_USE_FILE_LOCKING='FALSE' 
echo "Run a Python script"
python3 deep_nnEq_1t.py ${SLURM_ARRAY_TASK_ID}
echo "Ended batch processing at `date`."
