#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --partition=sfscai
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=hh3043@nyu.edu
#SBATCH --requeue
#SBATCH --nodelist=gpu190

source /gpfsnyu/packages/miniconda/2023.2.7/etc/profile.d/conda.sh
ml cuda/11.1.1
conda activate molbart
python LLM4Chem/generate_on_dataset.py --model_name osunlp/LlaSMol-Mistral-7B --output_dir eval/LlaSMol-Mistral-7B/output --tasks "['forward_synthesis','retrosynthesis']"
python LLM4Chem/extract_prediction.py --output_dir eval/LlaSMol-Mistral-7B/output --prediction_dir eval/LlaSMol-Mistral-7B/prediction --tasks "['forward_synthesis','retrosynthesis']"
conda deactivate