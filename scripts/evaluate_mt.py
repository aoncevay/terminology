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

models = ["MADLAD", "NLLB", "LLM.aya"] #, "LLM.tower"]
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
    "NLLB": "\\textsc{NLLB}",
    "MADLAD": "\\textsc{Madlad}",
    "LLM.aya": "\\textsc{Aya}",
    "LLM.tower": "\\textsc{Tower}"
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
                            stats_results[dataset][comparison][direction] = {
                                "p_value": p_value,
                                "mean_diff": mean_diff,
                                "confidence_interval": [ci_low, ci_high],
                                "significant": p_value < 0.05,
                                "better_model": model1 if mean_diff > 0 else model2
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
                if src == 'en' and tgt not in languages_in_data:
                    languages_in_data.append(tgt)
    
    languages_in_data = sorted(languages_in_data)
    
    # Table 1: English to XX - chrF++ scores
    print("\n% Table for English to Target Language - chrF++ scores")
    print("\\begin{table*}")
    print("\\centering")
    
    # Calculate column width
    num_models = len(models)
    col_spec = "|l|" + "c|" * num_models
    
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
            row = LANGID2LATEX.get(lang, lang)  # Use formatted language code
            
            # Find best chrF++ score for this language
            best_score = -1
            model_scores = {}
            
            # First pass: collect scores and find the best
            for model in models:
                lang_pair = f"en-{lang}"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["chrf++"] != -1:
                    score = results[model][lang_pair]["chrf++"]
                    model_scores[model] = score
                    if score > best_score:
                        best_score = score
            
            # Second pass: format output with bold for best score
            for model in models:
                lang_pair = f"en-{lang}"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["chrf++"] != -1:
                    score = model_scores[model]
                    if score == best_score:
                        row += f" & \\textbf{{{score:.2f}}}"
                    else:
                        row += f" & {score:.2f}"
                else:
                    row += " & -"
                    
            print(row + " \\\\")
    
    # Add average row only if we have languages with data
    if languages_with_data:
        avg_row = "\\hline\nAvg."
        
        # First pass for average: collect and find best
        model_avg_scores = {}
        best_avg = -1
        
        for model in models:
            chrf_values = []
            for lang in languages_with_data:
                lang_pair = f"en-{lang}"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["chrf++"] != -1:
                    chrf_values.append(results[model][lang_pair]["chrf++"])
            
            if chrf_values:
                avg_chrf = sum(chrf_values) / len(chrf_values)
                model_avg_scores[model] = avg_chrf
                if avg_chrf > best_avg:
                    best_avg = avg_chrf
        
        # Second pass for average: format with bold for best
        for model in models:
            if model in model_avg_scores:
                avg_score = model_avg_scores[model]
                if avg_score == best_avg:
                    avg_row += f" & \\textbf{{{avg_score:.2f}}}"
                else:
                    avg_row += f" & {avg_score:.2f}"
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
            
            # Find best model and its score for this language
            best_model = None
            best_score = -1
            available_models = []
            model_scores = {}
            
            # First, collect all available scores
            for model in models:
                lang_pair = f"en-{lang}"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["term_acc"] != -1:
                    score = results[model][lang_pair]["term_acc"]
                    model_scores[model] = score
                    available_models.append(model)
                    if score > best_score:
                        best_score = score
                        best_model = model
            
            # Term accuracy scores
            for model in models:
                lang_pair = f"en-{lang}"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["term_acc"] != -1:
                    # Add symbol for statistical significance if applicable
                    symbol = ""
                    if stats_results and model == best_model and len(available_models) > 1:
                        # Check if this model is significantly better than ANY other model
                        is_significant = False
                        for other_model in available_models:
                            if other_model == model:
                                continue
                            
                            comparison = f"{model}_vs_{other_model}"
                            alt_comparison = f"{other_model}_vs_{model}" 
                            
                            if comparison in stats_results.get(dataset, {}):
                                if lang_pair in stats_results[dataset][comparison]:
                                    test_result = stats_results[dataset][comparison][lang_pair]
                                    if test_result["significant"] and test_result["better_model"] == model:
                                        is_significant = True
                                        break
                            elif alt_comparison in stats_results.get(dataset, {}):
                                if lang_pair in stats_results[dataset][alt_comparison]:
                                    test_result = stats_results[dataset][alt_comparison][lang_pair]
                                    if test_result["significant"] and test_result["better_model"] == model:
                                        is_significant = True
                                        break
                        
                        if is_significant:
                            symbol = "$^\\dagger$"  # Significant improvement
                    
                    # Format with bold for best score
                    if model_scores[model] == best_score:
                        row += f" & \\textbf{{{model_scores[model]:.2f}}}{symbol}"
                    else:
                        row += f" & {model_scores[model]:.2f}{symbol}"
                else:
                    row += " & -"
                    
            print(row + " \\\\")
    
    # Add average row only if we have languages with data
    if languages_with_data:
        avg_row = "\\hline\nAvg."
        
        # First pass for average: collect and find best
        model_avg_scores = {}
        best_avg = -1
        
        for model in models:
            term_acc_values = []
            for lang in languages_with_data:
                lang_pair = f"en-{lang}"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["term_acc"] != -1:
                    term_acc_values.append(results[model][lang_pair]["term_acc"])
            
            if term_acc_values:
                avg_term_acc = sum(term_acc_values) / len(term_acc_values)
                model_avg_scores[model] = avg_term_acc
                if avg_term_acc > best_avg:
                    best_avg = avg_term_acc
        
        # Second pass for average: format with bold for best
        for model in models:
            if model in model_avg_scores:
                avg_score = model_avg_scores[model]
                if avg_score == best_avg:
                    avg_row += f" & \\textbf{{{avg_score:.2f}}}"
                else:
                    avg_row += f" & {avg_score:.2f}"
            else:
                avg_row += " & -"
        
        print(avg_row + " \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Term Accuracy Scores for " + dataset.upper() + " dataset (\\textsc{en}$\\rightarrow$\\textsc{xx}). $^\\dagger$ indicates statistically significant improvement (p < 0.05)}")
    print("\\label{tab:" + dataset + "-en-to-xx-term}")
    print("\\end{table*}")
    
    # Table 3: XX to English - chrF++ scores
    print("\n% Table for Target Language to English - chrF++ scores")
    print("\\begin{table*}")
    print("\\centering")
    
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
            
            # Find best chrF++ score for this language
            best_score = -1
            model_scores = {}
            
            # First pass: collect scores and find the best
            for model in models:
                lang_pair = f"{lang}-en"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["chrf++"] != -1:
                    score = results[model][lang_pair]["chrf++"]
                    model_scores[model] = score
                    if score > best_score:
                        best_score = score
            
            # Second pass: format output with bold for best score
            for model in models:
                lang_pair = f"{lang}-en"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["chrf++"] != -1:
                    score = model_scores[model]
                    if score == best_score:
                        row += f" & \\textbf{{{score:.2f}}}"
                    else:
                        row += f" & {score:.2f}"
                else:
                    row += " & -"
                    
            print(row + " \\\\")
        
    # Add average row only if we have languages with data
    if languages_with_data:
        avg_row = "\\hline\nAvg."
        
        # First pass for average: collect and find best
        model_avg_scores = {}
        best_avg = -1
        
        for model in models:
            chrf_values = []
            for lang in languages_with_data:
                lang_pair = f"{lang}-en"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["chrf++"] != -1:
                    chrf_values.append(results[model][lang_pair]["chrf++"])
            
            if chrf_values:
                avg_chrf = sum(chrf_values) / len(chrf_values)
                model_avg_scores[model] = avg_chrf
                if avg_chrf > best_avg:
                    best_avg = avg_chrf
        
        # Second pass for average: format with bold for best
        for model in models:
            if model in model_avg_scores:
                avg_score = model_avg_scores[model]
                if avg_score == best_avg:
                    avg_row += f" & \\textbf{{{avg_score:.2f}}}"
                else:
                    avg_row += f" & {avg_score:.2f}"
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
            
            # Find best model and its score for this language
            best_model = None
            best_score = -1
            available_models = []
            model_scores = {}
            
            # First, collect all available scores
            for model in models:
                lang_pair = f"{lang}-en"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["term_acc"] != -1:
                    score = results[model][lang_pair]["term_acc"]
                    model_scores[model] = score
                    available_models.append(model)
                    if score > best_score:
                        best_score = score
                        best_model = model
            
            # Term accuracy scores
            for model in models:
                lang_pair = f"{lang}-en"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["term_acc"] != -1:
                    # Add symbol for statistical significance if applicable
                    symbol = ""
                    if stats_results and model == best_model and len(available_models) > 1:
                        # Check if this model is significantly better than ANY other model
                        is_significant = False
                        for other_model in available_models:
                            if other_model == model:
                                continue
                            
                            comparison = f"{model}_vs_{other_model}"
                            alt_comparison = f"{other_model}_vs_{model}" 
                            
                            if comparison in stats_results.get(dataset, {}):
                                if lang_pair in stats_results[dataset][comparison]:
                                    test_result = stats_results[dataset][comparison][lang_pair]
                                    if test_result["significant"] and test_result["better_model"] == model:
                                        is_significant = True
                                        break
                            elif alt_comparison in stats_results.get(dataset, {}):
                                if lang_pair in stats_results[dataset][alt_comparison]:
                                    test_result = stats_results[dataset][alt_comparison][lang_pair]
                                    if test_result["significant"] and test_result["better_model"] == model:
                                        is_significant = True
                                        break
                        
                        if is_significant:
                            symbol = "$^\\dagger$"  # Significant improvement
                    
                    # Format with bold for best score
                    if model_scores[model] == best_score:
                        row += f" & \\textbf{{{model_scores[model]:.2f}}}{symbol}"
                    else:
                        row += f" & {model_scores[model]:.2f}{symbol}"
                else:
                    row += " & -"
                    
            print(row + " \\\\")
        
    # Add average row only if we have languages with data
    if languages_with_data:
        avg_row = "\\hline\nAvg."
        
        # First pass for average: collect and find best
        model_avg_scores = {}
        best_avg = -1
        
        for model in models:
            term_acc_values = []
            for lang in languages_with_data:
                lang_pair = f"{lang}-en"
                if model in results and lang_pair in results[model] and results[model][lang_pair]["term_acc"] != -1:
                    term_acc_values.append(results[model][lang_pair]["term_acc"])
            
            if term_acc_values:
                avg_term_acc = sum(term_acc_values) / len(term_acc_values)
                model_avg_scores[model] = avg_term_acc
                if avg_term_acc > best_avg:
                    best_avg = avg_term_acc
        
        # Second pass for average: format with bold for best
        for model in models:
            if model in model_avg_scores:
                avg_score = model_avg_scores[model]
                if avg_score == best_avg:
                    avg_row += f" & \\textbf{{{avg_score:.2f}}}"
                else:
                    avg_row += f" & {avg_score:.2f}"
            else:
                avg_row += " & -"
        
        print(avg_row + " \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Term Accuracy Scores for " + dataset.upper() + " dataset (\\textsc{xx}$\\rightarrow$\\textsc{en}). $^\\dagger$ indicates statistically significant improvement (p < 0.05)}")
    print("\\label{tab:" + dataset + "-xx-to-en-term}")
    print("\\end{table*}")
    
    # Reset stdout if we redirected it
    if output_file:
        sys.stdout = orig_stdout

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate MT models on terminology datasets")
    
    # Add argument group for execution control
    action_group = parser.add_argument_group('Actions')
    action_group.add_argument("--metrics", action="store_true", help="Compute evaluation metrics")
    action_group.add_argument("--stats", action="store_true", help="Run statistical significance tests")
    action_group.add_argument("--tables", action="store_true", help="Print LaTeX tables")
    
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

if __name__ == "__main__":
    main()
