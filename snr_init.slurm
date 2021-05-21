#!/bin/bash
#SBATCH -J snr_init # Name for your job
#SBATCH -n 1 # Number of tasks when using MPI. Default is 1, and SLURM assumes the usage of 1 cpu per task.
#SBATCH -N 1 # Number of nodes to spread cores across - default is 1 - if you are not using MPI this should likely be 1
#SBATCH --mem 6000 # Megabytes of memory requested. Default is 2000/task.
#SBATCH -t 0-00:15:00 # Runtime in days-hours:minutes:seconds.
#SBATCH -p express  # Partition to submit to the standard compute node partition(defq) or the express node partition(express)
#SBATCH --mail-user alexander.salois@student.montana.edu # this is the email you wish to be notified.
#SBATCH --mail-type FAIL,END # this specifies what events you should get an email about ALL will alert you of job beginning, completion, failure, etc.
#SBATCH -o snr_init_%j.out.txt # Standard output
#SBATCH -e snr_init_%j.err.txt # Standard error

date
echo "Hello from $(hostname)."

module load matlab/R2017a

echo "Run a matlab script"

matlab -nodesktop -nodisplay -nosplash -r "saveSNRdata; exit;"

echo "Ended batch processing at `date`."
