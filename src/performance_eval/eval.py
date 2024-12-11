import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import SmoothingFunction
import nltk
from nltk import word_tokenize
import os
from eval_functions import *

def main(args):
    # load the devset
    gold_answers = pd.read_csv(args.gold_answers).astype(str)['answer'].tolist()
    print(f"Loaded {len(gold_answers)} gold answers")

    # load all answers output/inference (.txt files separated by newlines)
    model_answers = {}

    # load answers from each model's output file
    for file in os.listdir(args.inference_dir):
        if file.endswith('.txt'):
            model_name = file.replace(".txt", "")
            with open(f"{args.inference_dir}/{file}", "r") as f:
                model_answers[model_name] = f.read().splitlines()
                print(f"Loaded answers from model: {model_name}")

    # evaluate the answers
    model_answers_list = list(model_answers.values())
    model_names = list(model_answers.keys())
    results = evaluate_answers(model_answers_list, gold_answers, model_names)
    print("\nEvaluation Results:")
    print(pd.DataFrame(results))

    # comparison plot
    if args.comparison_plot:
        comparison_plot(results, metrics=args.metrics, titles=args.titles, save_path=args.comparison_plot, retriever_order=args.retriever_order)
    
    # save the results to a csv file if specified
    if args.save_results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.save_results, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model answers against gold answers')
    
    parser.add_argument('--gold-answers', type=str, default="data/dev_set.csv",
                        help='Path to CSV file containing gold answers')
    parser.add_argument('--inference-dir', type=str, default="output",
                        help='Directory containing model inference output files')
    parser.add_argument('--comparison-plot', type=str,
                        help='Path to save comparison plot image')
    parser.add_argument('--save-results', type=str,
                        help='Path to save evaluation results CSV')
    parser.add_argument('--metrics', nargs='+', 
                        help='List of metrics to plot (e.g., BLEU ROUGE-1 ROUGE-2 ROUGE-L METEOR)')
    parser.add_argument('--titles', nargs='+',
                        help='Custom titles for the plots')
    parser.add_argument('--retriever-order', nargs='+',
                        help='Order of retrievers in the comparison plot')
    
    
    args = parser.parse_args()
    main(args)
