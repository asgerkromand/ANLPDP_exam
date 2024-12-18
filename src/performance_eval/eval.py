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
    print(results)
    
    # comparison plot
    if args.comparison_plot:
        comparison_plot(results, metrics=args.metrics, titles=args.titles, save_path=args.comparison_plot, retriever_order=args.retriever_order, retrievers=args.retrievers)

    # cls plot
    if args.cls_plot:
        cls_plot(results, metrics=args.cls_metrics, save_path=args.cls_save_path)

    # save the results to a csv file if specified
    if args.save_results:
        results_df = pd.DataFrame(results)
        results_df = results_df.reindex(sorted(results_df.columns), axis=1)
        results_df = results_df.drop(columns=['neo_gen_random_context', 't5_gen_random_context'])
        results_df.columns = results_df.columns.str.replace('gen_', '')
        # Sort the columns into to dataframes based on being either Neo or T5
        neo_cols = [col for col in results_df.columns if 'neo' in col]
        t5_cols = [col for col in results_df.columns if 't5' in col]
        neo_results = results_df[neo_cols]
        t5_results = results_df[t5_cols]
        # Remove neo and t5 from the column names
        neo_results.columns = neo_results.columns.str.replace('neo_', '')
        t5_results.columns = t5_results.columns.str.replace('t5_', '')
        # Make sure that baseline and upper bound is the two first columns
        neo_results = neo_results[['baseline', 'upper_bound'] + [col for col in neo_results.columns if col not in ['baseline', 'upper_bound']]]
        t5_results = t5_results[['baseline', 'upper_bound'] + [col for col in t5_results.columns if col not in ['baseline', 'upper_bound']]]
        # Save the results to a tex file with only two decimals
        neo_results.to_latex(
            f'{args.save_results}_neo.tex', 
            index=True, 
            float_format="%.2f") # Scores GPT-Neo 1.3B
        t5_results.to_latex(
            f'{args.save_results}_t5.tex', 
            index=True, 
            float_format="%.2f") # Scores DantT5-Large 770M
        results_df.to_latex(args.save_results.replace('all.csv', '.tex'), index=True, float_format="%.2f")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model answers against gold answers')
    
    parser.add_argument('--gold_answers', type=str, default="data/dev_set.csv",
                        help='Path to CSV file containing gold answers')
    parser.add_argument('--inference_dir', type=str, default="output/inference",
                        help='Directory containing model inference output files')
    parser.add_argument('--comparison_plot', type=str,
                        help='Path to save comparison plot image')
    parser.add_argument('--save_results', type=str,
                        help='Path to save evaluation results LaTeX')
    parser.add_argument('--metrics', nargs='+', 
                        help='List of metrics to plot (e.g., BLEU ROUGE-1 ROUGE-2 ROUGE-L METEOR)')
    parser.add_argument('--titles', nargs='+',
                        help='Custom titles for the plots')
    parser.add_argument('--retriever_order', nargs='+',
                        help='Order of retrievers in the comparison plot')
    parser.add_argument('--retrievers', nargs='+',
                        help='List of retrievers to include in the comparison plot')
    parser.add_argument('--cls_plot', type=str,
                        help='Path to save CLS plot image')
    parser.add_argument('--cls_metrics', nargs='+',
                        help='List of metrics to plot in CLS plot')
    parser.add_argument('--cls_save_path', type=str,
                        help='Path to save CLS plot image')
    
    args = parser.parse_args()
    main(args)
