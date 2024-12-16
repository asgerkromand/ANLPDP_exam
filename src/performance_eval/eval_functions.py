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

def comparison_plot(results, metrics=None, titles=None, save_path=None, figsize=(12, 8), retriever_order=None, retrievers=None):
 
    # get unique retriever names after stripping _k1, _k2, _k3 and model type prefix
    retriever_names = list(set(model.replace('_k1', '').replace('_k2', '').replace('_k3', '').replace('t5_gen_', '').replace('neo_gen_', '') for model in results.keys()))
    
    # remove the baselines and upper bounds from the retriever names
    retriever_names = [name for name in retriever_names if name not in ["baseline", "upper_bound"]]

    # Filter retrievers if specified
    if retrievers is not None:
        # Verify specified retrievers exist
        invalid_retrievers = set(retrievers) - set(retriever_names)
        if invalid_retrievers:
            raise ValueError(f"Invalid retrievers specified: {invalid_retrievers}")
        retriever_names = [r for r in retriever_names if r in retrievers]

    # Sort retrievers if order is provided
    if retriever_order is not None:
        # Verify all retrievers are present
        if set(retriever_order) != set(retriever_names):
            raise ValueError("retriever_order must contain all and only the retriever names present in results")
        retriever_names = retriever_order

    # there are two files called "t5_gen_random_contexts.txt" and "neo_gen_random_contexts.txt" - we want to ignore these
    results = {k: v for k, v in results.items() if k not in ["t5_gen_random_contexts", "neo_gen_random_contexts"]}

    # get baselines and upper bounds
    t5_baseline = results["t5_gen_baseline"]
    neo_baseline = results["neo_gen_baseline"]
    t5_upper = results["t5_gen_upper_bound"]
    neo_upper = results["neo_gen_upper_bound"]

    # Filter metrics if specified
    if metrics is not None:
        t5_baseline = {k: v for k, v in t5_baseline.items() if k in metrics}
        neo_baseline = {k: v for k, v in neo_baseline.items() if k in metrics}
        t5_upper = {k: v for k, v in t5_upper.items() if k in metrics}
        neo_upper = {k: v for k, v in neo_upper.items() if k in metrics}

    # Filter results based on specified metrics
    filtered_results = {}
    for model, scores in results.items():
        if metrics is not None:
            filtered_results[model] = {k: v for k, v in scores.items() if k in metrics}
        else:
            filtered_results[model] = scores
    results = filtered_results

    # y axis max value
    max_value = max(max(scores.values()) for scores in results.values())

    # set up plot (n_retrievers rows, 2 columns)
    n_retrievers = len(retriever_names)
    fig, axes = plt.subplots(n_retrievers, 2, figsize=figsize)
    if n_retrievers == 1:
        axes = axes.reshape(1, 2)

    # Add titles to the top plots
    axes[0,0].set_title("T5 Models")
    axes[0,1].set_title("Neo Models")

    # Create hatch pattern for upper bounds
    hatch_pattern = '//'

    # Define color shades for bars
    t5_colors = ['#1f77b4', '#4e97d1', '#7fb7ee', '#afd7ff', '#d4e8ff']  # Blue shades
    neo_colors = ['#1f77b4', '#4e97d1', '#7fb7ee', '#afd7ff', '#d4e8ff']  # Same blue shades

    # Plot each model type in its own panel with all k configurations
    for idx, retriever in enumerate(retriever_names):
        # Plot T5 results in left column
        ax_t5 = axes[idx, 0]
        model = f"t5_gen_{retriever}"
        x_positions = np.arange(len(results[f'{model}_k1'].keys()))
        width = 0.15  # Reduced width to create more space between categories
        
        # Plot bars for each k value side by side
        ax_t5.bar(x_positions - 2*width, t5_upper.values(), width, label='p*', color='white', edgecolor='black', hatch=hatch_pattern, alpha=0.5)
        ax_t5.bar(x_positions - width, t5_baseline.values(), width, label='p=0', color=t5_colors[1])
        ax_t5.bar(x_positions, results[f'{model}_k1'].values(), width, label='p=1', color=t5_colors[2])
        ax_t5.bar(x_positions + width, results[f'{model}_k2'].values(), width, label='p=2', color=t5_colors[3])
        ax_t5.bar(x_positions + 2*width, results[f'{model}_k3'].values(), width, label='p=3', color=t5_colors[4])
        
        ax_t5.set_ylabel(retriever)
        if idx == len(retriever_names) - 1:
            ax_t5.set_xticks(x_positions)
            ax_t5.set_xticklabels(results[f'{model}_k1'].keys())
        else:
            ax_t5.set_xticks([])
        ax_t5.set_ylim(0, max_value * 1.1)
        ax_t5.grid(axis='y', linestyle='--', alpha=0.3)
        if idx == 0:  # Only show legend for first plot
            ax_t5.legend()

        # Plot Neo results in right column
        ax_neo = axes[idx, 1]
        model = f"neo_gen_{retriever}"
        
        # Plot bars for each k value side by side
        ax_neo.bar(x_positions - 2*width, neo_upper.values(), width, label='p*', color='white', edgecolor='black', hatch=hatch_pattern, alpha=0.5)
        ax_neo.bar(x_positions - width, neo_baseline.values(), width, label='p=0', color=neo_colors[1])
        ax_neo.bar(x_positions, results[f'{model}_k1'].values(), width, label='p=1', color=neo_colors[2])
        ax_neo.bar(x_positions + width, results[f'{model}_k2'].values(), width, label='p=2', color=neo_colors[3])
        ax_neo.bar(x_positions + 2*width, results[f'{model}_k3'].values(), width, label='p=3', color=neo_colors[4])
        
        if idx == len(retriever_names) - 1:
            ax_neo.set_xticks(x_positions)
            ax_neo.set_xticklabels(results[f'{model}_k1'].keys())
        else:
            ax_neo.set_xticks([])
        ax_neo.set_ylim(0, max_value * 1.1)
        ax_neo.set_yticklabels([])
        ax_neo.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='svg')
    else:
        plt.show()
    return fig