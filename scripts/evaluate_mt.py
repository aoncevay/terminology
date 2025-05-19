import os
import json
import sacrebleu
import numpy as np
import random
import tempfile
import subprocess
import argparse
import sys
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

# Removed Task2_LLM variants
models = ["MADLAD", "NLLB", "LLM.aya", "LLM.llama", "LLM_mistral", "LLM_openai_gpt4o"] 
# Not including ++ variants to simplify the tables, and also, they are part of other analysis

datasets = ["irs", "cfpb"]
languages = ["es", "kr", "ru", "vi", "zh_s", "zh_t", "ht"]
chrf2 = sacrebleu.CHRF(word_order=2)

dataset2langs = {
    "irs": ["es", "kr", "ru", "vi", "zh_s", "zh_t", "ht"],
    "cfpb": ["es", "kr", "ru", "vi", "zh_t", "ht"]
}

# Language name mapping
LANG2NAME = {
    "en": "English",
    "es": "Spanish",
    "kr": "Korean",
    "ru": "Russian",
    "vi": "Vietnamese",
    "zh_s": "Chinese Simplified",
    "zh_t": "Chinese Traditional",
    "ht": "Haitian Creole"
}

MODELSNAME2LATEX = {
    # MT systems
    "NLLB": "\\textsc{NLLB}",
    "MADLAD": "\\textsc{Madlad}",
    # small LLMs
    "LLM.aya": "\\textsc{Aya23}",
    "LLM.tower": "\\textsc{Tower}",
    "LLM.llama": "\\textsc{Llama3.1}",
    "LLM_mistral": "\\textsc{Mistral}",
    # large LLM
    "LLM_openai_gpt4o": "\\textsc{GPT4o}"
    # Removed ++ variants
}

LANGID2LATEX = {
    "es": "\\textsc{es}",
    "kr": "\\textsc{ko}",
    "ru": "\\textsc{ru}",
    "vi": "\\textsc{vi}",
    "zh_s": "\\textsc{zh(s)}",
    "zh_t": "\\textsc{zh(t)}",
    "ht": "\\textsc{ht}"
}

# Bootstrap parameters
NUM_BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_LEVEL = 0.95  # 95% confidence interval

# Define model group information to help with formatting
MODEL_GROUPS = {
    # MT systems
    "NLLB": "mt",
    "MADLAD": "mt",
    # small LLMs
    "LLM.aya": "small_llm",
    "LLM.llama": "small_llm",
    "LLM_mistral": "small_llm",
    # large LLM
    "LLM_openai_gpt4o": "large_llm"
    # Removed prompt variant groups
}

# Remove prompt variant pairs
# Define which models to compare for the general MT vs small LLM comparison
MT_MODELS = ["NLLB", "MADLAD"]
SMALL_LLM_MODELS = ["LLM.aya", "LLM.llama", "LLM_mistral"]

# Combined group of comparable models (MT systems and small LLMs, excluding prompt variants)
COMPARABLE_MODELS = MT_MODELS + SMALL_LLM_MODELS

# Define baseline models (excluding prompt variants)
BASELINE_MODELS = ["MADLAD", "NLLB", "LLM.aya", "LLM.llama", "LLM_mistral", "LLM_openai_gpt4o"]

# Define colors for model groups
COLOR_MAP = {
    # MT systems - shades of blue
    "MADLAD": "#1f77b4",  # dark blue
    "NLLB": "#7fdbff",    # light blue
    
    # Small LLMs - shades of green
    "LLM.aya": "#2ca02c",     # dark green
    "LLM.llama": "#98df8a",   # medium green
    "LLM_mistral": "#d4ffaa", # light green
    
    # Large LLM - purple instead of red for better contrast with mean line
    "LLM_openai_gpt4o": "#9467bd"  # purple
}

def evaluate_all_datasets():
    """Evaluate all models and datasets, saving results per model"""
    # Dictionary to hold all results
    results_scores = {}  # Stores average scores
    results_values = {}  # Stores raw term accuracy values (0/1) for bootstrap
    
    # Initialize result dictionaries
    for model in models:
        results_scores[model] = {}
        results_values[model] = {}
        for dataset in datasets:
            results_scores[model][dataset] = {}
            results_values[model][dataset] = {}
    
    # Evaluate each model
    for model in models:
        print(f"Evaluating {model} model")
        for dataset in datasets:
            print(f"  Evaluating {dataset} dataset")
            for language in languages:
                # Evaluate both directions
                for lang_pair in [f"{language}-en", f"en-{language}"]:
                    results_scores[model][dataset][lang_pair] = {}
                    results_values[model][dataset][lang_pair] = {}
                    
                    output_path = f"../output/{model}.{dataset}.{lang_pair}.json"
                    output_path_v2 = f"../output/{model}_{dataset}_{lang_pair}.json"
                    if os.path.exists(output_path) or os.path.exists(output_path_v2):
                        if os.path.exists(output_path):
                            with open(output_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                        else:
                            with open(output_path_v2, "r", encoding="utf-8") as f:
                                data = json.load(f)

                        # Compute chrF++ score
                        try:
                            chrf = chrf2.corpus_score(data["hyp"], [data["ref"]])
                            print(f"    chrF++ score for {lang_pair}: {chrf.score}")
                            results_scores[model][dataset][lang_pair]["chrf++"] = chrf.score
                        except Exception as e:
                            print(f"    Error computing chrF++ score for {lang_pair}: {e}")
                            results_scores[model][dataset][lang_pair]["chrf++"] = -1
                            results_scores[model][dataset][lang_pair]["term_acc"] = -1
                            results_values[model][dataset][lang_pair]["term_acc"] = []
                            continue
                            
                        # Compute term accuracy and store binary results
                        term_accuracy = 0
                        term_counter = 0
                        term_acc_values = []  # Binary success/failure for each term
                        
                        for term_pair, hyp_translation in zip(data["term_pairs"], data["hyp"]):
                            # Track results for each individual term
                            sentence_results = []
                            for k, v in term_pair.items():
                                found = v.lower() in hyp_translation.lower()
                                sentence_results.append(1 if found else 0)
                                term_accuracy += 1 if found else 0
                                term_counter += 1
                            
                            if sentence_results:
                                term_acc_values.append(sentence_results)
                        
                        # Calculate average term accuracy
                        term_accuracy_score = term_accuracy / term_counter if term_counter > 0 else 0
                        print(f"    Term accuracy for {lang_pair}: {term_accuracy_score}")
                        
                        # Store both the overall score and the binary results
                        results_scores[model][dataset][lang_pair]["term_acc"] = term_accuracy_score
                        results_values[model][dataset][lang_pair]["term_acc"] = term_acc_values
                    else:
                        print(f"    No output file found for {model}.{dataset}.{lang_pair}")
                        results_scores[model][dataset][lang_pair]["chrf++"] = -1
                        results_scores[model][dataset][lang_pair]["term_acc"] = -1
                        results_values[model][dataset][lang_pair]["term_acc"] = []

        # Save results per model
        with open(f"../results/scores_{model}.json", "w", encoding="utf-8") as f:
            json.dump(results_scores[model], f, indent=4)
        with open(f"../results/values_{model}.json", "w", encoding="utf-8") as f:
            json.dump(results_values[model], f, indent=4)
    
    # Return results for further analysis
    return results_scores, results_values

def load_results():
    """Load previously computed results"""
    results_scores = {}
    results_values = {}
    
    for model in models:
        scores_path = f"../results/scores_{model}.json"
        values_path = f"../results/values_{model}.json"
        
        if os.path.exists(scores_path):
            with open(scores_path, "r", encoding="utf-8") as f:
                results_scores[model] = json.load(f)
        
        if os.path.exists(values_path):
            with open(values_path, "r", encoding="utf-8") as f:
                results_values[model] = json.load(f)
    
    return results_scores, results_values

def bootstrap_term_accuracy(values1, values2):
    """
    Perform bootstrap resampling to determine if the difference between
    two models' term accuracy results is statistically significant.
    
    Args:
        values1: Binary term accuracy results for model 1 (list of lists)
        values2: Binary term accuracy results for model 2 (list of lists)
    
    Returns:
        p_value: p-value of the difference
        mean_diff: mean difference between the two models
        ci_low: lower bound of confidence interval
        ci_high: upper bound of confidence interval
    """
    if not values1 or not values2:
        return None, None, None, None
    
    # Flatten lists if needed while preserving equal weighting of sentences
    def compute_accuracy_from_binary(values):
        if not values:
            return 0
        total_correct = sum(sum(sentence) for sentence in values)
        total_terms = sum(len(sentence) for sentence in values)
        return total_correct / total_terms if total_terms > 0 else 0
    
    # Original accuracies
    acc1 = compute_accuracy_from_binary(values1)
    acc2 = compute_accuracy_from_binary(values2)
    mean_diff = acc1 - acc2
    
    # Initialize bootstrap differences
    bootstrap_diffs = []
    
    # Ensure we have the same number of samples from each model
    n = min(len(values1), len(values2))
    if n == 0:
        return None, None, None, None
        
    # Create paired samples
    for _ in range(NUM_BOOTSTRAP_SAMPLES):
        # Sample with replacement
        indices = [random.randint(0, n-1) for _ in range(n)]
        
        # Compute accuracies for the bootstrap sample
        sample1 = [values1[i] for i in indices if i < len(values1)]
        sample2 = [values2[i] for i in indices if i < len(values2)]
        
        bootstrap_acc1 = compute_accuracy_from_binary(sample1)
        bootstrap_acc2 = compute_accuracy_from_binary(sample2)
        
        # Record the difference
        bootstrap_diffs.append(bootstrap_acc1 - bootstrap_acc2)
    
    # Sort differences for percentile calculation
    bootstrap_diffs.sort()
    
    # Calculate confidence interval
    alpha = 1 - CONFIDENCE_LEVEL
    lower_idx = int(NUM_BOOTSTRAP_SAMPLES * (alpha / 2))
    upper_idx = int(NUM_BOOTSTRAP_SAMPLES * (1 - alpha / 2))
    ci_low = bootstrap_diffs[lower_idx]
    ci_high = bootstrap_diffs[upper_idx]
    
    # Calculate p-value (two-tailed test)
    # Count how many bootstrap samples have the opposite sign of the original difference
    opposite_sign = sum(1 for diff in bootstrap_diffs if diff * mean_diff <= 0)
    p_value = opposite_sign / NUM_BOOTSTRAP_SAMPLES
    
    return p_value, mean_diff, ci_low, ci_high

def run_statistical_tests(results_values):
    """Run statistical tests between pairs of models for each dataset and language pair"""
    stats_results = {}
    
    # For each dataset and language pair, compare each pair of models
    for dataset in datasets:
        stats_results[dataset] = {}
        
        for i, model1 in enumerate(models):
            if model1 not in results_values or dataset not in results_values[model1]:
                continue
                
            for j, model2 in enumerate(models):
                if i == j or model2 not in results_values or dataset not in results_values[model2]:
                    continue
                    
                # Skip if both models are not in COMPARABLE_MODELS (skip GPT4o comparisons)
                if model1 not in COMPARABLE_MODELS and model2 not in COMPARABLE_MODELS:
                    continue
                    
                comparison = f"{model1}_vs_{model2}"
                stats_results[dataset][comparison] = {}
                
                # Compare for each language pair
                for language in languages:
                    for direction in [f"{language}-en", f"en-{language}"]:
                        # Skip if any data is missing
                        if (direction not in results_values[model1][dataset] or 
                            direction not in results_values[model2][dataset] or
                            "term_acc" not in results_values[model1][dataset][direction] or
                            "term_acc" not in results_values[model2][dataset][direction]):
                            continue
                        
                        # Get term accuracy values for both models
                        values1 = results_values[model1][dataset][direction]["term_acc"]
                        values2 = results_values[model2][dataset][direction]["term_acc"]
                        
                        # Run bootstrap test
                        p_value, mean_diff, ci_low, ci_high = bootstrap_term_accuracy(values1, values2)
                        
                        if p_value is not None:
                            # Determine the comparison type - simplify to just mt_vs_llm
                            comparison_type = "general"
                            
                            # Check if this is an MT vs small LLM comparison
                            if (model1 in MT_MODELS and model2 in SMALL_LLM_MODELS) or \
                               (model1 in SMALL_LLM_MODELS and model2 in MT_MODELS):
                                comparison_type = "mt_vs_llm"
                            
                            stats_results[dataset][comparison][direction] = {
                                "p_value": p_value,
                                "mean_diff": mean_diff,
                                "confidence_interval": [ci_low, ci_high],
                                "significant": p_value < 0.05,
                                "better_model": model1 if mean_diff > 0 else model2,
                                "comparison_type": comparison_type
                            }
    
    # Save statistical test results
    with open(f"../results/statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(stats_results, f, indent=4)
    
    return stats_results

def load_statistical_tests():
    """Load previously computed statistical test results"""
    stats_path = "../results/statistical_tests.json"
    
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    return None

def print_table_latex_per_dataset(results_scores, stats_results=None, dataset="irs", output_file=None):
    """Print LaTeX tables for a given dataset, including statistical significance if provided"""
    # Use file output if specified
    if output_file:
        orig_stdout = sys.stdout
        sys.stdout = output_file
    
    # Build dictionary for easier access
    results = {}
    for model in models:
        if model in results_scores and dataset in results_scores[model]:
            results[model] = results_scores[model][dataset]
    
    # Get languages for this dataset from dataset2langs
    languages_in_data = dataset2langs.get(dataset, [])
    if not languages_in_data:
        # Fallback to extracting languages from results if dataset not in dataset2langs
        for model in results:
            for lang_pair in results[model]:
                src, tgt = lang_pair.split('-')
                if src == 'en' and tgt not in languages_in_data and tgt != 'en':
                    languages_in_data.append(tgt)
    
    languages_in_data = sorted(languages_in_data)
    
    # Helper function to determine if a model has the best score in its group
    def is_best_in_group(model, group_models, lang_pair, metric):
        best_score = -1
        best_model = None
        
        for m in group_models:
            if m in results and lang_pair in results[m] and results[m][lang_pair][metric] != -1:
                score = results[m][lang_pair][metric]
                if score > best_score:
                    best_score = score
                    best_model = m
        
        return model == best_model
    
    # Helper function to determine if a model is significantly better than all others in its group
    def is_significantly_better_than_second_best(model, group_models, lang_pair, stats_results):
        if not stats_results:
            return False
            
        # Get scores for all models in the group
        models_with_scores = []
        for m in group_models:
            if m in results and lang_pair in results[m] and results[m][lang_pair]["term_acc"] != -1:
                models_with_scores.append((m, results[m][lang_pair]["term_acc"]))
        
        # Sort by score in descending order
        models_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # If our model is not the best or there's only one model, return False
        if not models_with_scores or models_with_scores[0][0] != model or len(models_with_scores) < 2:
            return False
        
        # Check if the best model is significantly better than the second best
        second_best = models_with_scores[1][0]
        
        # Determine which comparison to look for
        comparison = f"{model}_vs_{second_best}"
        alt_comparison = f"{second_best}_vs_{model}"
        
        # Check the first direction
        if comparison in stats_results.get(dataset, {}):
            if lang_pair in stats_results[dataset][comparison]:
                test_result = stats_results[dataset][comparison][lang_pair]
                if test_result["significant"] and test_result["better_model"] == model:
                    return True
        
        # Check the opposite direction
        elif alt_comparison in stats_results.get(dataset, {}):
            if lang_pair in stats_results[dataset][alt_comparison]:
                test_result = stats_results[dataset][alt_comparison][lang_pair]
                if test_result["significant"] and test_result["better_model"] == model:
                    return True
        
        return False

    # Table 1: English to XX - chrF++ scores
    print("\n% Table for English to Target Language - chrF++ scores")
    print("\\begin{table*}")
    print("\\centering")
    
    # Update the column specification - with double vertical bars for formatting
    col_spec = "|l||c|c||c|c|c||c|"
    
    print("\\begin{tabular}{" + col_spec + "}")
    print("\\hline")
    
    # Header row
    header = "Lang."
    for model in models:
        header += " & " + MODELSNAME2LATEX.get(model, model)
    print(header + " \\\\")
    print("\\hline")
    
    # Data rows - only include languages that have data
    languages_with_data = []
    for lang in languages_in_data:
        has_data = False
        for model in models:
            lang_pair = f"en-{lang}"
            if model in results and lang_pair in results[model] and results[model][lang_pair]["chrf++"] != -1:
                has_data = True
                break
        
        if has_data:
            languages_with_data.append(lang)
            row = LANGID2LATEX.get(lang, lang)  # Use formatted language code - remove the extra dash after this
            
            # chrF++ scores - no bold formatting
            for model in models:
                lang_pair = f"en-{lang}"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["chrf++"] != -1:
                    score = results[model][lang_pair]["chrf++"]
                    row += f" & {score:.2f}"
                else:
                    row += " & -"
        
            print(row + " \\\\")
    
    # Add average row only if we have languages with data
    if languages_with_data:
        avg_row = "\\hline\nAvg."
        
        # Calculate average scores without bold formatting
        for model in models:
            chrf_values = []
            for lang in languages_with_data:
                lang_pair = f"en-{lang}"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["chrf++"] != -1:
                    chrf_values.append(results[model][lang_pair]["chrf++"])
            
            # Fixed indentation here to ensure proper averaging
            if chrf_values:
                avg_chrf = sum(chrf_values) / len(chrf_values)
                avg_row += f" & {avg_chrf:.2f}"
            else:
                avg_row += " & -"
                
        print(avg_row + " \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{chrF++ Scores for " + dataset.upper() + " dataset (\\textsc{en}$\\rightarrow$\\textsc{xx})}")
    print("\\label{tab:" + dataset + "-en-to-xx-chrf}")
    print("\\end{table*}")
    
    # Table 2: English to XX - Term Accuracy scores
    print("\n% Table for English to Target Language - Term Accuracy scores")
    print("\\begin{table*}")
    print("\\centering")
    
    # Use the same column separator pattern for all tables
    print("\\begin{tabular}{" + col_spec + "}")
    print("\\hline")
    
    # Header row
    header = "Lang."
    for model in models:
        header += " & " + MODELSNAME2LATEX.get(model, model)
    print(header + " \\\\")
    print("\\hline")
    
    # Data rows - only include languages that have data
    languages_with_data = []
    for lang in languages_in_data:
        has_data = False
        for model in models:
            lang_pair = f"en-{lang}"
            if model in results and lang_pair in results[model] and results[model][lang_pair]["term_acc"] != -1:
                has_data = True
                break
        
        if has_data:
            languages_with_data.append(lang)
            row = LANGID2LATEX.get(lang, lang)  # Use formatted language code
            
            # Get comparable models with data for this language (MT systems and small LLMs, excluding prompt variants)
            comparable_models_with_data = [m for m in COMPARABLE_MODELS if m in results and lang_pair in results[m] and results[m][lang_pair]["term_acc"] != -1]
            
            # Term accuracy scores with statistical significance markers
            for model in models:
                lang_pair = f"en-{lang}"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["term_acc"] != -1:
                    score = results[model][lang_pair]["term_acc"]
                    
                    # Check for statistical significance markers
                    best_model_symbol = ""
                    
                    if stats_results:
                        # Check if this model is the best among comparable models (MT and small LLMs) and is significantly better
                        # Only add asterisk for models in the comparable group 
                        if model in COMPARABLE_MODELS and comparable_models_with_data:
                            if is_best_in_group(model, comparable_models_with_data, lang_pair, "term_acc") and \
                               is_significantly_better_than_second_best(model, comparable_models_with_data, lang_pair, stats_results):
                                best_model_symbol = "$^*$"  # Asterisk for best model among MT and small LLMs
                    
                    # Format without bold, add significance symbols if applicable
                    row += f" & {score:.2f}{best_model_symbol}"
                else:
                    row += " & -"
                
            print(row + " \\\\")
        
    # Add average row only if we have languages with data
    if languages_with_data:
        avg_row = "\\hline\nAvg."
        
        # Calculate average scores for each model
        model_avg_scores = {}
        
        for model in models:
            term_acc_values = []
            for lang in languages_with_data:
                lang_pair = f"en-{lang}"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["term_acc"] != -1:
                    term_acc_values.append(results[model][lang_pair]["term_acc"])
            
            if term_acc_values:
                avg_term_acc = sum(term_acc_values) / len(term_acc_values)
                model_avg_scores[model] = avg_term_acc
        
        # Get comparable models with average scores
        comparable_models_with_avg = [m for m in COMPARABLE_MODELS if m in model_avg_scores]
        
        # Helper function to check if a model is significantly better on average
        def is_significantly_better_on_average(model, other_model, stats_results):
            if not stats_results:
                return False
                
            # Check significance for each language
            significant_wins = 0
            for lang in languages_with_data:
                lang_pair = f"en-{lang}"
                comparison = f"{model}_vs_{other_model}"
                alt_comparison = f"{other_model}_vs_{model}"
                
                if comparison in stats_results.get(dataset, {}) and lang_pair in stats_results[dataset][comparison]:
                    test_result = stats_results[dataset][comparison][lang_pair]
                    if test_result["significant"] and test_result["better_model"] == model:
                        significant_wins += 1
                
                elif alt_comparison in stats_results.get(dataset, {}) and lang_pair in stats_results[dataset][alt_comparison]:
                    test_result = stats_results[dataset][alt_comparison][lang_pair]
                    if test_result["significant"] and test_result["better_model"] == model:
                        significant_wins += 1
            
            # If model wins significantly for at least one language, consider it better on average
            return significant_wins > 0
        
        # Format average row with significance markers
        for model in models:
            if model in model_avg_scores:
                avg_score = model_avg_scores[model]
                
                # Check for statistical significance in average
                best_model_symbol = ""
                
                if stats_results:
                    # Check for best model among comparable models significance in average
                    if model in COMPARABLE_MODELS and comparable_models_with_avg:
                        if model == max(comparable_models_with_avg, key=lambda m: model_avg_scores[m]) and len(comparable_models_with_avg) > 1:
                            # Find the second-best model
                            second_best = sorted(comparable_models_with_avg, key=lambda m: model_avg_scores[m])[-2]
                            if is_significantly_better_on_average(model, second_best, stats_results):
                                best_model_symbol = "$^*$"  # Asterisk for best model among MT and small LLMs
                
                # Format without bold, add significance symbols
                avg_row += f" & {avg_score:.2f}{best_model_symbol}"
            else:
                avg_row += " & -"
        
        print(avg_row + " \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Term Accuracy Scores for " + dataset.upper() + " dataset (\\textsc{en}$\\rightarrow$\\textsc{xx}). $^*$ indicates significant improvement over other comparable models (MT systems and small LLMs)}")
    print("\\label{tab:" + dataset + "-en-to-xx-term}")
    print("\\end{table*}")
    
    # Table 3: XX to English - chrF++ scores
    print("\n% Table for Target Language to English - chrF++ scores")
    print("\\begin{table*}")
    print("\\centering")
    
    # Use the same column separator pattern for all tables
    print("\\begin{tabular}{" + col_spec + "}")
    print("\\hline")
    
    # Header row
    header = "Lang."
    for model in models:
        header += " & " + MODELSNAME2LATEX.get(model, model)
    print(header + " \\\\")
    print("\\hline")
    
    # Data rows - only include languages that have data
    languages_with_data = []
    for lang in languages_in_data:
        has_data = False
        for model in models:
            lang_pair = f"{lang}-en"
            if model in results and lang_pair in results[model] and results[model][lang_pair]["chrf++"] != -1:
                has_data = True
                break
        
        if has_data:
            languages_with_data.append(lang)
            row = LANGID2LATEX.get(lang, lang)  # Use formatted language code
            
            # chrF++ scores - no bold formatting
            for model in models:
                lang_pair = f"{lang}-en"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["chrf++"] != -1:
                    score = results[model][lang_pair]["chrf++"]
                    row += f" & {score:.2f}"
                else:
                    row += " & -"
                    
            print(row + " \\\\")
        
    # Add average row only if we have languages with data
    if languages_with_data:
        avg_row = "\\hline\nAvg."
        
        # Calculate average scores without bold formatting
        for model in models:
            chrf_values = []
            for lang in languages_with_data:
                lang_pair = f"{lang}-en"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["chrf++"] != -1:
                    chrf_values.append(results[model][lang_pair]["chrf++"])
            
            if chrf_values:
                avg_chrf = sum(chrf_values) / len(chrf_values)
                avg_row += f" & {avg_chrf:.2f}"
            else:
                avg_row += " & -"
        
        print(avg_row + " \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{chrF++ Scores for " + dataset.upper() + " dataset (\\textsc{xx}$\\rightarrow$\\textsc{en})}")
    print("\\label{tab:" + dataset + "-xx-to-en-chrf}")
    print("\\end{table*}")
    
    # Table 4: XX to English - Term Accuracy scores
    print("\n% Table for Target Language to English - Term Accuracy scores")
    print("\\begin{table*}")
    print("\\centering")
    
    # Use the same column separator pattern for all tables
    print("\\begin{tabular}{" + col_spec + "}")
    print("\\hline")
    
    # Header row
    header = "Lang."
    for model in models:
        header += " & " + MODELSNAME2LATEX.get(model, model)
    print(header + " \\\\")
    print("\\hline")
    
    # Data rows - only include languages that have data
    languages_with_data = []
    for lang in languages_in_data:
        has_data = False
        for model in models:
            lang_pair = f"{lang}-en"
            if model in results and lang_pair in results[model] and results[model][lang_pair]["term_acc"] != -1:
                has_data = True
                break
        
        if has_data:
            languages_with_data.append(lang)
            row = LANGID2LATEX.get(lang, lang)  # Use formatted language code
            
            # Get comparable models with data for this language (MT systems and small LLMs, excluding prompt variants)
            comparable_models_with_data = [m for m in COMPARABLE_MODELS if m in results and lang_pair in results[m] and results[m][lang_pair]["term_acc"] != -1]
            
            # Term accuracy scores with statistical significance markers
            for model in models:
                lang_pair = f"{lang}-en"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["term_acc"] != -1:
                    score = results[model][lang_pair]["term_acc"]
                    
                    # Check for statistical significance markers
                    best_model_symbol = ""
                    
                    if stats_results:
                        # Check if this model is the best among comparable models (MT and small LLMs) and is significantly better
                        # Only add asterisk for models in the comparable group 
                        if model in COMPARABLE_MODELS and comparable_models_with_data:
                            if is_best_in_group(model, comparable_models_with_data, lang_pair, "term_acc") and \
                               is_significantly_better_than_second_best(model, comparable_models_with_data, lang_pair, stats_results):
                                best_model_symbol = "$^*$"  # Asterisk for best model among MT and small LLMs
                    
                    # Format without bold, add significance symbols if applicable
                    row += f" & {score:.2f}{best_model_symbol}"
                else:
                    row += " & -"
                    
            print(row + " \\\\")
        
    # Add average row only if we have languages with data
    if languages_with_data:
        avg_row = "\\hline\nAvg."
        
        # Calculate average scores for each model
        model_avg_scores = {}
        
        for model in models:
            term_acc_values = []
            for lang in languages_with_data:
                lang_pair = f"{lang}-en"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["term_acc"] != -1:
                    term_acc_values.append(results[model][lang_pair]["term_acc"])
            
            if term_acc_values:
                avg_term_acc = sum(term_acc_values) / len(term_acc_values)
                model_avg_scores[model] = avg_term_acc
        
        # Get comparable models with average scores
        comparable_models_with_avg = [m for m in COMPARABLE_MODELS if m in model_avg_scores]
        
        # Format average row with significance markers
        for model in models:
            if model in model_avg_scores:
                avg_score = model_avg_scores[model]
                
                # Check for statistical significance in average
                best_model_symbol = ""
                
                if stats_results:
                    # Check for best model among comparable models significance in average
                    if model in COMPARABLE_MODELS and comparable_models_with_avg:
                        if model == max(comparable_models_with_avg, key=lambda m: model_avg_scores[m]) and len(comparable_models_with_avg) > 1:
                            # Find the second-best model
                            second_best = sorted(comparable_models_with_avg, key=lambda m: model_avg_scores[m])[-2]
                            if is_significantly_better_on_average(model, second_best, stats_results):
                                best_model_symbol = "$^*$"  # Asterisk for best model among MT and small LLMs
                
                # Format without bold, add significance symbols
                avg_row += f" & {avg_score:.2f}{best_model_symbol}"
            else:
                avg_row += " & -"
        
        print(avg_row + " \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Term Accuracy Scores for " + dataset.upper() + " dataset (\\textsc{xx}$\\rightarrow$\\textsc{en}). $^*$ indicates significant improvement over other comparable models (MT systems and small LLMs)}")
    print("\\label{tab:" + dataset + "-xx-to-en-term}")
    print("\\end{table*}")
    
    # Reset stdout if we redirected it
    if output_file:
        sys.stdout = orig_stdout

def create_boxplots(results_scores, stats_results=None, figs_dir="../figs"):
    """Create boxplots for the baseline models for each dataset, translation direction, and metric"""
    # Create figs directory if it doesn't exist
    os.makedirs(figs_dir, exist_ok=True)
    
    # Set matplotlib parameters for a clean, publication-ready style
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12
    
    # List to track all generated figures for LaTeX
    figure_info = []
    
    # Process each dataset
    for dataset in datasets:
        # Process each metric
        for metric in ["chrf++", "term_acc"]:
            # Create list to track figures for this dataset-metric combination
            dataset_metric_figures = []
            
            # Process each translation direction
            for direction in ["en-xx", "xx-en"]:
                # Collect data for all models and languages
                data_by_model = {}
                available_langs = set()
                
                for model in BASELINE_MODELS:
                    if model not in results_scores:
                        continue
                        
                    if dataset not in results_scores[model]:
                        continue
                    
                    scores = []
                    langs = []
                    
                    # Get the languages for this dataset
                    dataset_langs = dataset2langs.get(dataset, [])
                    
                    for lang in dataset_langs:
                        if direction == "en-xx":
                            lang_pair = f"en-{lang}"
                        else:
                            lang_pair = f"{lang}-en"
                            
                        if lang_pair in results_scores[model][dataset] and results_scores[model][dataset][lang_pair][metric] != -1:
                            scores.append(results_scores[model][dataset][lang_pair][metric])
                            langs.append(lang)
                            available_langs.add(lang)
                    
                    if scores:  # Only include model if it has data
                        # Store scores with their corresponding languages
                        scores_with_langs = list(zip(scores, langs))
                        data_by_model[model] = {
                            "scores": scores,
                            "langs": langs,
                            "scores_with_langs": scores_with_langs
                        }
                
                # Skip if no data
                if not data_by_model:
                    continue
                
                # Create a more compact figure
                fig, ax = plt.subplots(figsize=(7, 3))
                
                # Prepare data for boxplot
                boxplot_data = []
                model_names = []
                colors = []
                best_langs = []  # Track best lang for each model
                worst_langs = []  # Track worst lang for each model
                
                for model in BASELINE_MODELS:
                    if model in data_by_model:
                        boxplot_data.append(data_by_model[model]["scores"])
                        # Clean model names (remove LaTeX formatting)
                        model_names.append(MODELSNAME2LATEX.get(model, model).replace("\\textsc{", "").replace("}", ""))
                        colors.append(COLOR_MAP.get(model, "gray"))
                        
                        # Find best and worst languages for this model
                        scores_with_langs = data_by_model[model]["scores_with_langs"]
                        best_score_with_lang = max(scores_with_langs, key=lambda x: x[0])
                        worst_score_with_lang = min(scores_with_langs, key=lambda x: x[0])
                        
                        best_langs.append((best_score_with_lang[0], best_score_with_lang[1]))
                        worst_langs.append((worst_score_with_lang[0], worst_score_with_lang[1]))
                
                # Create the boxplot with thicker lines for mean
                boxplot = ax.boxplot(boxplot_data, patch_artist=True, widths=0.5, meanprops={'linewidth': 2})
                
                # Apply colors to boxes with higher alpha for better visibility
                for box, color in zip(boxplot['boxes'], colors):
                    box.set_facecolor(color)
                    box.set_alpha(0.7)
                
                # Make median lines more visible
                for median in boxplot['medians']:
                    median.set_linewidth(2.0)
                    median.set_color('black')
                
                # Get flier (outlier) positions for all models
                outliers_by_position = {}
                for i, fliers in enumerate(boxplot['fliers']):
                    if len(fliers.get_data()[1]) > 0:  # If there are outliers
                        outliers_by_position[i+1] = fliers.get_data()[1]  # Store outlier y-positions
                
                # Get lower whisker positions for clash detection
                whisker_ends = {}
                for i, whisker in enumerate(boxplot['whiskers']):
                    # Whiskers come in pairs (upper and lower)
                    if i % 2 == 1:  # Lower whisker (odd indices)
                        box_pos = (i // 2) + 1
                        whisker_data = whisker.get_data()
                        whisker_ends[box_pos] = whisker_data[1][1]  # Y-coordinate of whisker end
                
                # Add language labels for best and worst performers
                for i, ((best_score, best_lang), (worst_score, worst_lang)) in enumerate(zip(best_langs, worst_langs)):
                    # Position calculations - slightly offset from the whiskers
                    box_pos = i + 1  # Boxplot positions are 1-indexed
                    
                    # Calculate smaller offset based on the y-axis range (closer to box limits)
                    y_offset = 1.0 if metric == "chrf++" else 0.01
                    
                    # Add best language with font size matching axis labels
                    ax.text(box_pos, best_score + y_offset, 
                           LANGID2LATEX.get(best_lang, best_lang).replace("\\textsc{", "").replace("}", ""),
                           horizontalalignment='center', verticalalignment='bottom', 
                           fontsize=12, fontstyle='italic')
                    
                    # Check if the worst score is an outlier
                    is_outlier = False
                    if box_pos in outliers_by_position:
                        outlier_values = outliers_by_position[box_pos]
                        # If the worst score is very close to any outlier value, consider it an outlier
                        if any(abs(worst_score - outlier) < 1e-5 for outlier in outlier_values):
                            is_outlier = True
                    
                    # For outliers, default to placing labels below the point
                    # Only place above if there's a large gap between whisker and outlier
                    if is_outlier:
                        # Always place labels below outliers by default
                        place_above = False
                        
                        # Only consider placing above if there's a significant gap
                        if box_pos in whisker_ends:
                            whisker_end = whisker_ends[box_pos]
                            # Only place above if outlier is very far from whisker (more than 25% of the plotting range)
                            gap_factor = 0.25
                            if metric == "chrf++":
                                # For chrF++, y-range is 0-85
                                min_gap_needed = 85 * gap_factor
                            else:
                                # For term_acc, y-range is 0-1.0
                                min_gap_needed = 1.0 * gap_factor
                                
                            if (whisker_end - worst_score) > min_gap_needed:
                                place_above = True
                        
                        if place_above:
                            # Place label above the outlier point
                            ax.text(box_pos, worst_score + y_offset, 
                                   LANGID2LATEX.get(worst_lang, worst_lang).replace("\\textsc{", "").replace("}", ""),
                                   horizontalalignment='center', verticalalignment='bottom', 
                                   fontsize=12, fontstyle='italic')
                        else:
                            # Place label below as usual
                            ax.text(box_pos, worst_score - y_offset, 
                                   LANGID2LATEX.get(worst_lang, worst_lang).replace("\\textsc{", "").replace("}", ""),
                                   horizontalalignment='center', verticalalignment='top', 
                                   fontsize=12, fontstyle='italic')
                    else:
                        # Normal case - place label below
                        ax.text(box_pos, worst_score - y_offset, 
                               LANGID2LATEX.get(worst_lang, worst_lang).replace("\\textsc{", "").replace("}", ""),
                               horizontalalignment='center', verticalalignment='top', 
                               fontsize=12, fontstyle='italic')
                
                # Add asterisks for statistical significance ONLY for term_acc metric
                has_significance = False
                if stats_results and metric == "term_acc":
                    # Check if model is significantly better than others
                    for i, model in enumerate(BASELINE_MODELS):
                        if model not in data_by_model:
                            continue
                            
                        # Check if model is the best in its group and significantly better
                        model_in_comparable = model in COMPARABLE_MODELS
                        is_best = False
                        
                        # Check if it's the best on average
                        if model_in_comparable:
                            avg_scores = {m: sum(data_by_model[m]["scores"]) / len(data_by_model[m]["scores"]) 
                                        for m in data_by_model if m in COMPARABLE_MODELS}
                            if avg_scores and model in avg_scores:
                                is_best = model == max(avg_scores, key=avg_scores.get)
                        
                        # Check if it's significantly better than at least one other model
                        is_significant = False
                        if model_in_comparable and is_best:
                            for other_model in COMPARABLE_MODELS:
                                if other_model == model or other_model not in data_by_model:
                                    continue
                                    
                                # Check significant wins for each language
                                for lang in available_langs:
                                    if direction == "en-xx":
                                        lang_pair = f"en-{lang}"
                                    else:
                                        lang_pair = f"{lang}-en"
                                    
                                    # Check both directions of comparison
                                    comparison = f"{model}_vs_{other_model}"
                                    alt_comparison = f"{other_model}_vs_{model}"
                                    
                                    if dataset in stats_results:
                                        if comparison in stats_results[dataset] and lang_pair in stats_results[dataset][comparison]:
                                            test_result = stats_results[dataset][comparison][lang_pair]
                                            if test_result["significant"] and test_result["better_model"] == model:
                                                is_significant = True
                                                has_significance = True
                                                break
                                        
                                        elif alt_comparison in stats_results[dataset] and lang_pair in stats_results[dataset][alt_comparison]:
                                            test_result = stats_results[dataset][alt_comparison][lang_pair]
                                            if test_result["significant"] and test_result["better_model"] == model:
                                                is_significant = True
                                                has_significance = True
                                                break
                                
                                if is_significant:
                                    break
                        
                        # Position asterisk above the best language label
                        if is_significant:
                            best_score = best_langs[i][0]
                            # Use consistent offset for asterisk (above the language label)
                            asterisk_offset = 3.0 if metric == "chrf++" else 0.03
                            ax.text(i + 1, best_score + asterisk_offset, "*",
                                   horizontalalignment='center', fontsize=20, fontweight='bold')
                
                # Set only necessary labels
                metric_name = "chrF++" if metric == "chrf++" else "Term Accuracy"
                
                # Set only the y-axis label
                ax.set_ylabel(metric_name, fontsize=14)
                
                # Set x-tick labels horizontal (no rotation)
                ax.set_xticklabels(model_names, fontsize=12)
                
                # Set fixed y-axis limits based on metric
                if metric == "chrf++":
                    plt.ylim(0, 85)  # Fixed range for chrF++ plots
                else:  # term_acc
                    plt.ylim(0, 1.0)  # Fixed range for term accuracy plots
                
                # Adjust layout and save
                plt.tight_layout()
                
                # Save the figure
                filename = f"{dataset}_{direction}_{metric}_boxplot.pdf"
                filepath = os.path.join(figs_dir, filename)
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                print(f"Saved boxplot to {filepath}")
                
                # Store figure info for LaTeX
                direction_label = "EN→XX" if direction == "en-xx" else "XX→EN"
                dataset_metric_figures.append({
                    "path": filename,
                    "direction": direction_label,
                    "direction_code": direction,
                    "has_significance": has_significance
                })
                
                plt.close()
            
            # Only add to figure_info if we have both directions
            if len(dataset_metric_figures) == 2:
                figure_info.append({
                    "dataset": dataset.upper(),
                    "metric": metric,
                    "figures": dataset_metric_figures
                })
    
    # Generate LaTeX file with all figures
    generate_latex_figures(figure_info, figs_dir)
    
    return figure_info

def generate_latex_figures(figure_info, figs_dir):
    """Generate LaTeX file with all boxplot figures"""
    
    latex_file = os.path.join(figs_dir, "boxplots.tex")
    
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write("% Boxplots for Translation Evaluation\n")
        f.write("% This file was automatically generated\n\n")
        
        # Add preamble with required packages
        f.write("% Required packages - include these in your main LaTeX document\n")
        f.write("% \\usepackage{graphicx}   % Required for images\n")
        f.write("% \\usepackage{subcaption} % Required for subfigures\n")
        f.write("% \\usepackage{float}      % Optional for better figure placement\n\n")
        
        # Generate a figure for each dataset-metric combination
        for i, info in enumerate(figure_info):
            dataset = info["dataset"]
            metric = info["metric"]
            figures = info["figures"]
            
            # Start figure environment
            f.write("\\begin{figure*}[t]\n")
            f.write("\\centering\n")
            
            # Subfigures for each direction
            for subfig in figures:
                path = subfig["path"]
                direction = subfig["direction"]
                subfigure_label = "a" if subfig["direction_code"] == "en-xx" else "b"
                
                # Use subcaption's subfigure environment instead of the deprecated subfigure package
                f.write(f"\\begin{{subfigure}}[b]{{0.48\\textwidth}}\n")
                f.write(f"    \\centering\n")
                f.write(f"    \\includegraphics[width=\\textwidth]{{{path}}}\n")
                f.write(f"    \\caption{{{direction}}}\n")
                f.write(f"    \\label{{fig:{dataset.lower()}_{metric}_{subfig['direction_code']}}}\n")
                f.write(f"\\end{{subfigure}}\n")
                f.write(f"\\hfill\n" if subfigure_label == "a" else "")
            
            # Caption and label for the entire figure
            metric_name = "chrF++" if metric == "chrf++" else "Term Accuracy"
            significance_note = ""
            if metric == "term_acc" and any(subfig["has_significance"] for subfig in figures):
                significance_note = " The asterisk (*) indicates models that are significantly better than other comparable models."
            
            caption = f"{metric_name} scores for {dataset} subset.{significance_note} Language labels at the top and bottom of each boxplot indicate the best and worst performing languages for each model, respectively."
            label = f"fig:{dataset.lower()}_{metric}"
            
            f.write(f"\\caption{{{caption}}}\n")
            f.write(f"\\label{{{label}}}\n")
            f.write("\\end{figure*}\n\n")
        
        # Add a note about usage at the end
        f.write("% Usage in main document:\n")
        f.write("% 1. Make sure to include the required packages above\n")
        f.write("% 2. Include this file with \\input{path/to/boxplots.tex}\n")
        
        print(f"LaTeX file with boxplots saved to {latex_file}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate MT models on terminology datasets")
    
    # Add argument group for execution control
    action_group = parser.add_argument_group('Actions')
    action_group.add_argument("--metrics", action="store_true", help="Compute evaluation metrics")
    action_group.add_argument("--stats", action="store_true", help="Run statistical significance tests")
    action_group.add_argument("--tables", action="store_true", help="Print LaTeX tables")
    action_group.add_argument("--plot_baseline", action="store_true", help="Create boxplots for baseline models")
    
    # Dataset selection
    parser.add_argument("--dataset", choices=datasets + ["all"], default="all", 
                        help="Dataset to evaluate (default: all)")
    
    # If no arguments are provided, show help
    if os.sys.argv[1:] == []:
        parser.print_help()
        exit(1)
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    results_scores = None
    results_values = None
    stats_results = None
    
    # Compute evaluation metrics if requested
    if args.metrics:
        print("Computing evaluation metrics...")
        results_scores, results_values = evaluate_all_datasets()
    
    # Run statistical tests if requested
    if args.stats:
        print("Running statistical significance tests...")
        if results_values is None:
            results_scores, results_values = load_results()
        stats_results = run_statistical_tests(results_values)
    
    # Print LaTeX tables if requested
    if args.tables:
        print("Generating LaTeX tables...")
        if results_scores is None:
            results_scores, results_values = load_results()
        if stats_results is None:
            stats_results = load_statistical_tests()
        
        # Determine which datasets to process
        datasets_to_process = [args.dataset] if args.dataset != "all" else datasets
        
        # Create output directory if it doesn't exist
        os.makedirs("../results/tables", exist_ok=True)
        
        for dataset in datasets_to_process:
            if dataset in datasets:  # Skip if dataset not valid
                output_filepath = f"../results/tables/{dataset}_tables.tex"
                with open(output_filepath, "w") as output_file:
                    print(f"Writing tables for {dataset.upper()} dataset to {output_filepath}")
                    print_table_latex_per_dataset(results_scores, stats_results, dataset=dataset, output_file=output_file)
                
                print(f"Tables for {dataset.upper()} dataset written to {output_filepath}")
    
    # Create boxplots if requested
    if args.plot_baseline:
        print("Creating boxplots for baseline models...")
        if results_scores is None:
            results_scores, results_values = load_results()
        if stats_results is None:
            stats_results = load_statistical_tests()
        
        create_boxplots(results_scores, stats_results)

if __name__ == "__main__":
    main()
