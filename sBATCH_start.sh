#! /bin/bash

# this is an example batch script for submitting a gpu job on the cluster
# first, we need to specify some variables for slurm, which is done via the SBATCH comments

#SBATCH -A kreshuk                             # specify the group
#SBATCH -N 1				       # specify the number of cluster nodes for the job
#SBATCH -n 3				       # specify the number of cores per node for the job
#SBATCH --mem 15G			       # specify the amount of memory per node
#SBATCH -t 5-0:10                              # specify the runtime of the job IMPORTANT: your job will get killed if it exceeds this runtime (the format is d-h:mm-ss)
#SBATCH -o outfile.out			       # specify the file to write the command line output to
#SBATCH -e errfile.err			       # specify the file to write the error output to
#SBATCH --mail-type=BEGIN,END,FAIL		       # specify mail notifications for your job 
#SBATCH --mail-user=alex.matskevych@embl.de    # specify the mail address for mail notifications 
#SBATCH -p gpu				       # specify the queue you want to submit to; here we choose the gpu queue. If you want to submit a pure CPU job, just leave this out.
#SBATCH -C gpu=1080Ti			       # specify the type of gpu you want
#SBATCH --gres=gpu:1			       # specify the number of gpus per node

# next we should load all the modules we need to run the job.
# in this example, I just load cuDNN, which pulls in all necessary CUDA dependencies
module load cuDNN

# finally, your script goes here
./train.py /g/kreshuk/matskevych/boundary_map_prediction/project_folder/ affinities_with_trafo
