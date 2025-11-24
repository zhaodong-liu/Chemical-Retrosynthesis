#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --partition=tandon_h100_1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=72:00:00
#SBATCH --account=pr_133_tandon_advanced
#SBATCH --gres=gpu:1      
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=hh3043@nyu.edu
#SBATCH --requeue


source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate phyre
python generate_on_dataset.py --model_name osunlp/LlaSMol-Mistral-7B --output_dir eval/LlaSMol-Mistral-7B/output --tasks "['retrosynthesis']"
python extract_prediction.py --output_dir eval/LlaSMol-Mistral-7B/output --prediction_dir eval/LlaSMol-Mistral-7B/prediction --tasks "['retrosynthesis']"
conda deactivate