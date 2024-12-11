import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import SmoothingFunction
import nltk
#nltk.download('wordnet')


# eval functions
def calculate_bleu(answers, gold_answers):
    scores = []
    for answer, gold_answer in zip(answers, gold_answers):
        score = sentence_bleu([gold_answer], answer, smoothing_function=SmoothingFunction().method1)
        scores.append(score)
    return scores

def calculate_rouge(answers, gold_answers):
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for answer, gold_answer in zip(answers, gold_answers):
        scores = scorer.score(answer, gold_answer)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
        
    return rouge1_scores, rouge2_scores, rougeL_scores

def calculate_meteor(answers, gold_answers):
    scores = []
    for answer, gold_answer in zip(answers, gold_answers):
        answer_tokens = answer.split()
        gold_answer_tokens = gold_answer.split()
        score = meteor_score([gold_answer_tokens], answer_tokens, gamma=0.0) # set gamma (word order penalty) to 0 to avoid bug where two identical sentences are not scored 1
        scores.append(score)
    return scores

# avg scores
def calculate_avg_scores(answers, gold_answers):
    bleu_scores = calculate_bleu(answers, gold_answers)
    rouge1_scores, rouge2_scores, rougeL_scores = calculate_rouge(answers, gold_answers)
    meteor_scores = calculate_meteor(answers, gold_answers)
    return np.mean(bleu_scores), np.mean(rouge1_scores), np.mean(rouge2_scores), np.mean(rougeL_scores), np.mean(meteor_scores)

# function to evalualate answers from different models
def evaluate_answers(model_answers_list, gold_answers, model_names=None):
    """
    Evaluates multiple models' answers against gold answers using various metrics
    
    Args:
        model_answers_list: List of lists, where each inner list contains answers from one model
        gold_answers: List of gold/reference answers
        model_names: Optional list of model names corresponding to the answers. If not provided,
                    defaults to "model_1", "model_2", etc.
    
    Returns:
        Dictionary of scores for each model
    """
    results = {}
    for i, answers in enumerate(model_answers_list):
        bleu_avg, rouge1_avg, rouge2_avg, rougeL_avg, meteor_avg = calculate_avg_scores(answers, gold_answers)
        model_key = model_names[i] if model_names and i < len(model_names) else f'model_{i+1}'
        results[model_key] = {
            'BLEU': bleu_avg,
            'ROUGE-1': rouge1_avg, 
            'ROUGE-2': rouge2_avg,
            'ROUGE-L': rougeL_avg,
            'METEOR': meteor_avg
        }
    return results

def plot_model_scores(results, metrics=None, titles=None, save_path=None):

    # if metrics is None, use all available metrics
    if metrics is None:
        metrics = list(next(iter(results.values())).keys())
    
    filtered_results = {
        model: {metric: score for metric, score in scores.items() if metric in metrics}
        for model, scores in results.items()
    }

    # y axis max value
    max_value = max(max(scores.values()) for scores in filtered_results.values())

    # grid dimensions
    n_models = len(filtered_results)
    n_cols = int(np.ceil(np.sqrt(n_models)))
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
    axes = axes.ravel()

    # plot each model
    for idx, (model, scores) in enumerate(filtered_results.items()):
        ax = axes[idx]
        ax.bar(scores.keys(), scores.values())
        title = titles[idx] if titles and idx < len(titles) else f'Model {model}'
        ax.set_title(title)
        ax.tick_params(axis='x')
        ax.set_ylim(0, max_value * 1.1)
    
    # hide any empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    return fig

def comparison_plot(results, metrics=None, titles=None, save_path=None, figsize=(15, 12)):
 
    # set up plot (5 rows)
    fig, axes = plt.subplots(5, 1, figsize=figsize)
    axes = axes.ravel()

    # get unique model names after stripping _k1, _k2, _k3
    model_names = list(set(model.replace('_k1', '').replace('_k2', '').replace('_k3', '') for model in results.keys()))

    # y axis max value
    max_value = max(max(scores.values()) for scores in results.values())

    # Plot each model type in its own panel with all k configurations
    for idx, model in enumerate(model_names):
        ax = axes[idx]
        x_positions = np.arange(len(results[f'{model}_k1'].keys()))
        width = 0.25  # Width of bars
        
        # Plot bars for each k value side by side
        ax.bar(x_positions - width, results[f'{model}_k1'].values(), width, label='p=1')
        ax.bar(x_positions, results[f'{model}_k2'].values(), width, label='p=2')
        ax.bar(x_positions + width, results[f'{model}_k3'].values(), width, label='p=3')
        
        ax.set_title(f'{model}')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(results[f'{model}_k1'].keys())
        ax.tick_params(axis='x')
        ax.set_ylim(0, max_value * 1.1)
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return fig