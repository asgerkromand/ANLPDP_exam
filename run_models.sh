#!/bin/bash

# Change directory to src/model_inference
cd /Users/asgerkromand/deep/ANLPDP_exam/src/model_inference || exit

# Run the commands and track progress

commands=(
  "python neo_generation.py --retriever tfidf --k_retrievals 1"
  "python neo_generation.py --retriever bm25 --k_retrievals 1"
  "python neo_generation.py --retriever bert_cls --k_retrievals 1"
  "python neo_generation.py --retriever bert_mean --k_retrievals 1"
  "python neo_generation.py --retriever bert_max --k_retrievals 1"
  "python t5_generation.py --retriever tfidf --k_retrievals 1"
  "python t5_generation.py --retriever bm25 --k_retrievals 1"
  "python t5_generation.py --retriever bert_cls --k_retrievals 1"
  "python t5_generation.py --retriever bert_mean --k_retrievals 1"
  "python t5_generation.py --retriever bert_max --k_retrievals 1"
)

for i in "${!commands[@]}"; do
  echo "Running command $((i+1))/${#commands[@]}: ${commands[$i]}"
  eval "${commands[$i]}"
  if [ $? -ne 0 ]; then
    echo "Command $((i+1)) failed. Exiting."
    exit 1
  fi
  echo "Command $((i+1)) completed successfully."
  echo "------------------------------"
done

# Use these two commandline code in terminal to run this file: 
# chmod +x run_models.sh
# ./run_models.sh

