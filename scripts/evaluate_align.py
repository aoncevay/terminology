models = ["mistral", "gpt"] # "llama"
datasets = ["irs", "cfpb"]
languages = ["es", "kr", "ru", "vi", "zh_s", "zh_t", "ht"]


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
    "llama": "\\textsc{Llama3.1}",
    "gpt": "\\textsc{GPT4o}",
    "mistral": "\\textsc{Mistral}"
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

import os
import json
import random
import argparse
import sys
from pathlib import Path
import numpy as np
from scipy import stats


def evaluate_all_datasets():
    """Evaluate all models and datasets for term pair extraction, saving results per model"""
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
                # Evaluate only en->language direction
                direction = f"en-{language}"
                file_prefix = f"{model}_{dataset}_{language}"
                
                results_scores[model][dataset][direction] = {}
                results_values[model][dataset][direction] = {}
                    
                output_path = f"../output_align/{model}/{file_prefix}_translations.json"
                if os.path.exists(output_path):
                    with open(output_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Process the data structure - flatten the document-level entries
                    flattened_entries = []
                    for doc_id, entries in data.items():
                        for entry in entries:
                            flattened_entries.append(entry)
                                
                    # Compute term accuracy and store binary results
                    term_accuracy = 0
                    term_counter = 0
                    term_acc_values = []  # Binary success/failure for each term pair
                    
                    for entry in flattened_entries:
                        # Check if all necessary fields are present
                        if "term_pairs" not in entry or "predicted_term_pairs" not in entry:
                            continue
                            
                        # Compare gold term pairs with predicted term pairs
                        gold_pairs = entry["term_pairs"]
                        pred_pairs = entry["predicted_term_pairs"]
                        entry_doc_id = entry.get("doc_id", "unknown")
                        
                        # Track results for each term pair in this entry
                        entry_results = []
                        
                        for source_term, gold_translation in gold_pairs.items():
                            if source_term in pred_pairs:
                                # Check if prediction is None
                                pred_translation = pred_pairs[source_term]
                                if pred_translation is None:
                                    # Log the issue and count as wrong
                                    print(f"      WARNING: None value in prediction for dataset={dataset}, language={language}, doc_id={entry_doc_id}, term={source_term}")
                                    entry_results.append(0)
                                    term_counter += 1
                                else:
                                    # Case-insensitive comparison
                                    try:
                                        correct = gold_translation.lower() == pred_translation.lower()
                                        entry_results.append(1 if correct else 0)
                                        term_accuracy += 1 if correct else 0
                                        term_counter += 1
                                    except AttributeError:
                                        # Handle any other unexpected types
                                        print(f"      WARNING: Type error comparing gold={type(gold_translation)} and pred={type(pred_translation)} for dataset={dataset}, language={language}, doc_id={entry_doc_id}, term={source_term}")
                                        print(f"      Gold: {gold_translation}, Pred: {pred_translation}")
                                        entry_results.append(0)
                                        term_counter += 1
                            else:
                                # Missing term in predictions - log and count as wrong
                                print(f"      WARNING: Missing term in prediction for dataset={dataset}, language={language}, doc_id={entry_doc_id}, term={source_term}")
                                entry_results.append(0)
                                term_counter += 1
                        
                        if entry_results:
                            term_acc_values.append(entry_results)
                    
                    # Calculate average term accuracy
                    term_accuracy_score = term_accuracy / term_counter if term_counter > 0 else 0
                    print(f"    Term accuracy for {language} (en->xx): {term_accuracy_score}")
                    
                    # Store both the overall score and the binary results
                    results_scores[model][dataset][direction]["term_acc"] = term_accuracy_score
                    results_values[model][dataset][direction]["term_acc"] = term_acc_values
                else:
                    print(f"    No output file found for {file_prefix}")
                    results_scores[model][dataset][direction]["term_acc"] = -1
                    results_values[model][dataset][direction]["term_acc"] = []

        # Save results per model
        os.makedirs("../results_align", exist_ok=True)
        with open(f"../results_align/scores_{model}.json", "w", encoding="utf-8") as f:
            json.dump(results_scores[model], f, indent=4)
        with open(f"../results_align/values_{model}.json", "w", encoding="utf-8") as f:
            json.dump(results_values[model], f, indent=4)
    
    # Return results for further analysis
    return results_scores, results_values


def load_results():
    """Load previously computed results"""
    results_scores = {}
    results_values = {}
    
    for model in models:
        scores_path = f"../results_align/scores_{model}.json"
        values_path = f"../results_align/values_{model}.json"
        
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
    
    # Flatten lists if needed while preserving equal weighting of entries
    def compute_accuracy_from_binary(values):
        if not values:
            return 0
        total_correct = sum(sum(entry) for entry in values)
        total_terms = sum(len(entry) for entry in values)
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
    
    # For each dataset, compare each pair of models
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
                
                # Compare for each language pair (only en->lang direction)
                for language in languages:
                    lang_pair = f"en-{language}"
                    
                    # Skip if any data is missing
                    if (lang_pair not in results_values[model1][dataset] or 
                        lang_pair not in results_values[model2][dataset] or
                        "term_acc" not in results_values[model1][dataset][lang_pair] or
                        "term_acc" not in results_values[model2][dataset][lang_pair]):
                        continue
                    
                    # Get term accuracy values for both models
                    values1 = results_values[model1][dataset][lang_pair]["term_acc"]
                    values2 = results_values[model2][dataset][lang_pair]["term_acc"]
                    
                    # Run bootstrap test
                    p_value, mean_diff, ci_low, ci_high = bootstrap_term_accuracy(values1, values2)
                    
                    if p_value is not None:
                        stats_results[dataset][comparison][lang_pair] = {
                            "p_value": p_value,
                            "mean_diff": mean_diff,
                            "confidence_interval": [ci_low, ci_high],
                            "significant": p_value < 0.05,
                            "better_model": model1 if mean_diff > 0 else model2
                        }
    
    # Save statistical test results
    os.makedirs("../results_align", exist_ok=True)
    with open(f"../results_align/statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(stats_results, f, indent=4)
    
    return stats_results


def load_statistical_tests():
    """Load previously computed statistical test results"""
    stats_path = "../results_align/statistical_tests.json"
    
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
    
    # Table: Term Alignment Accuracy (en->xx)
    print("\n% Table for Term Alignment Accuracy (en->xx)")
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
        
        # Check if language has data
        for model in models:
            direction = f"en-{lang}"
            if model in results and direction in results[model] and results[model][direction]["term_acc"] != -1:
                has_data = True
                break
        
        if has_data:
            languages_with_data.append(lang)
            row = LANGID2LATEX.get(lang, lang)  # Use formatted language code
            
            # Direction: en->xx
            direction = f"en-{lang}"
            
            # Find best model and score
            best_model = None
            best_score = -1
            available_models = []
            model_scores = {}
            
            for model in models:
                if model in results and direction in results[model] and results[model][direction]["term_acc"] != -1:
                    score = results[model][direction]["term_acc"]
                    model_scores[model] = score
                    available_models.append(model)
                    if score > best_score:
                        best_score = score
                        best_model = model
            
            # Add scores
            for model in models:
                if model in results and direction in results[model] and results[model][direction]["term_acc"] != -1:
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
                                if direction in stats_results[dataset][comparison]:
                                    test_result = stats_results[dataset][comparison][direction]
                                    if test_result["significant"] and test_result["better_model"] == model:
                                        is_significant = True
                                        break
                            elif alt_comparison in stats_results.get(dataset, {}):
                                if direction in stats_results[dataset][alt_comparison]:
                                    test_result = stats_results[dataset][alt_comparison][direction]
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
                direction = f"en-{lang}"
                if model in results and direction in results[model] and results[model][direction]["term_acc"] != -1:
                    term_acc_values.append(results[model][direction]["term_acc"])
            
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
    print("\\caption{Term Pair Extraction Accuracy for " + dataset.upper() + " dataset (\\textsc{en}$\\rightarrow$\\textsc{xx}). $^\\dagger$ indicates statistically significant improvement (p < 0.05)}")
    print("\\label{tab:" + dataset + "-term-align-accuracy}")
    print("\\end{table*}")
    
    # Reset stdout if we redirected it
    if output_file:
        sys.stdout = orig_stdout


def print_combined_table_latex(results_scores, stats_results=None, output_file=None):
    """Print a combined LaTeX table for all datasets, with IRS and CFPB side by side"""
    # Use file output if specified
    if output_file:
        orig_stdout = sys.stdout
        sys.stdout = output_file
    
    # Build dictionary for easier access
    results = {}
    for model in models:
        results[model] = {}
        for dataset in datasets:
            if model in results_scores and dataset in results_scores[model]:
                results[model][dataset] = results_scores[model][dataset]
    
    # Table: Term Alignment Accuracy (en->xx) - Combined datasets
    print("\n% Combined Table for Term Alignment Accuracy (en->xx)")
    print("\\begin{table}")
    print("\\centering")
    
    # Calculate column width for each dataset section - only one language column for both datasets
    num_models = len(models)
    # One column for language, then columns for models from both datasets
    col_spec = "|l|" + "c|" * num_models + "c|" * num_models
    
    print("\\begin{tabular}{" + col_spec + "}")
    print("\\hline")
    
    # Multi-column header for dataset names
    print("\\multicolumn{1}{|c|}{} & " + 
          "\\multicolumn{" + str(num_models) + "}{c|}{\\textbf{IRS}} & " + 
          "\\multicolumn{" + str(num_models) + "}{c|}{\\textbf{CFPB}} \\\\")
    print("\\hline")
    
    # Header row with model names (repeated for both datasets)
    header = "Lang."
    for model in models:
        header += " & " + MODELSNAME2LATEX.get(model, model)
    for model in models:
        header += " & " + MODELSNAME2LATEX.get(model, model)
    print(header + " \\\\")
    print("\\hline")
    
    # Pre-process to get all languages with data for each dataset
    dataset_languages = {}
    for dataset in datasets:
        languages_in_data = dataset2langs.get(dataset, [])
        if not languages_in_data:
            # Fallback to extracting languages from results if dataset not in dataset2langs
            for model in results:
                if dataset in results[model]:
                    for lang_pair in results[model][dataset]:
                        src, tgt = lang_pair.split('-')
                        if src == 'en' and tgt not in languages_in_data and tgt != 'en':
                            languages_in_data.append(tgt)
        
        # Only include languages that have data
        dataset_languages[dataset] = []
        for lang in sorted(languages_in_data):
            has_data = False
            for model in models:
                if dataset in results[model]:
                    direction = f"en-{lang}"
                    if direction in results[model][dataset] and results[model][dataset][direction]["term_acc"] != -1:
                        has_data = True
                        break
            if has_data:
                dataset_languages[dataset].append(lang)
    
    # Get all unique languages across both datasets
    all_languages = set()
    for dataset in datasets:
        all_languages.update(dataset_languages.get(dataset, []))
    all_languages = sorted(all_languages)
    
    # Build rows with data from both datasets side by side, one row per language
    for lang in all_languages:
        row = LANGID2LATEX.get(lang, lang)  # Language column
        
        # Process each dataset
        for dataset in datasets:
            # Check if this language has data for this dataset
            if lang in dataset_languages.get(dataset, []):
                direction = f"en-{lang}"
                
                # Find best model and score for statistical significance only
                best_model = None
                best_score = -1
                available_models = []
                model_scores = {}
                
                for model in models:
                    if dataset in results[model] and direction in results[model][dataset] and results[model][dataset][direction]["term_acc"] != -1:
                        score = results[model][dataset][direction]["term_acc"]
                        model_scores[model] = score
                        available_models.append(model)
                        if score > best_score:
                            best_score = score
                            best_model = model
                
                # Add scores
                for model in models:
                    if dataset in results[model] and direction in results[model][dataset] and results[model][dataset][direction]["term_acc"] != -1:
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
                                
                                if dataset in stats_results and comparison in stats_results[dataset]:
                                    if direction in stats_results[dataset][comparison]:
                                        test_result = stats_results[dataset][comparison][direction]
                                        if test_result["significant"] and test_result["better_model"] == model:
                                            is_significant = True
                                            break
                                elif dataset in stats_results and alt_comparison in stats_results[dataset]:
                                    if direction in stats_results[dataset][alt_comparison]:
                                        test_result = stats_results[dataset][alt_comparison][direction]
                                        if test_result["significant"] and test_result["better_model"] == model:
                                            is_significant = True
                                            break
                            
                            if is_significant:
                                symbol = "$^\\dagger$"  # Significant improvement
                        
                        # Format without bold, add significance symbol if applicable
                        row += f" & {model_scores[model]:.2f}{symbol}"
                    else:
                        row += " & -"
            else:
                # No data for this language in this dataset
                row += " & -" * num_models
        
        print(row + " \\\\")
    
    print("\\hline")
    
    # Add average row
    avg_row = "Avg."
    
    # Process average for each dataset
    for dataset in datasets:
        languages_with_data = dataset_languages.get(dataset, [])
        
        # First pass for average: collect scores and find best for significance
        model_avg_scores = {}
        best_avg = -1
        best_model = None
        
        for model in models:
            term_acc_values = []
            for lang in languages_with_data:
                direction = f"en-{lang}"
                if dataset in results[model] and direction in results[model][dataset] and results[model][dataset][direction]["term_acc"] != -1:
                    term_acc_values.append(results[model][dataset][direction]["term_acc"])
            
            if term_acc_values:
                avg_term_acc = sum(term_acc_values) / len(term_acc_values)
                model_avg_scores[model] = avg_term_acc
                if avg_term_acc > best_avg:
                    best_avg = avg_term_acc
                    best_model = model
        
        # Second pass for average: format averages
        for model in models:
            if model in model_avg_scores:
                avg_score = model_avg_scores[model]
                
                # Add significance symbol if this is the best model and significantly better than others
                symbol = ""
                if stats_results and model == best_model and len(model_avg_scores) > 1:
                    # Check if this model is significantly better than ANY other model
                    # For average row, we'll check all language pairs in this dataset
                    is_significant = False
                    for other_model in model_avg_scores.keys():
                        if other_model == model:
                            continue
                        
                        # Check if this model is significantly better than other model for any language
                        for lang in languages_with_data:
                            direction = f"en-{lang}"
                            comparison = f"{model}_vs_{other_model}"
                            alt_comparison = f"{other_model}_vs_{model}" 
                            
                            if dataset in stats_results and comparison in stats_results[dataset]:
                                if direction in stats_results[dataset][comparison]:
                                    test_result = stats_results[dataset][comparison][direction]
                                    if test_result["significant"] and test_result["better_model"] == model:
                                        is_significant = True
                                        break
                            elif dataset in stats_results and alt_comparison in stats_results[dataset]:
                                if direction in stats_results[dataset][alt_comparison]:
                                    test_result = stats_results[dataset][alt_comparison][direction]
                                    if test_result["significant"] and test_result["better_model"] == model:
                                        is_significant = True
                                        break
                        
                        if is_significant:
                            break
                    
                    if is_significant:
                        symbol = "$^\\dagger$"  # Significant improvement
                
                # No bold, just add symbol if applicable
                avg_row += f" & {avg_score:.2f}{symbol}"
            else:
                avg_row += " & -"
    
    print(avg_row + " \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Term Pair Extraction Accuracy for IRS and CFPB datasets (\\textsc{en}$\\rightarrow$\\textsc{xx}). $^\\dagger$ indicates statistically significant improvement (p < 0.05)}")
    print("\\label{tab:combined-term-align-accuracy}")
    print("\\end{table}")
    
    # Reset stdout if we redirected it
    if output_file:
        sys.stdout = orig_stdout


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate alignment models on terminology datasets")
    
    # Add argument group for execution control
    action_group = parser.add_argument_group('Actions')
    action_group.add_argument("--metrics", action="store_true", help="Compute evaluation metrics")
    action_group.add_argument("--stats", action="store_true", help="Run statistical significance tests")
    action_group.add_argument("--tables", action="store_true", help="Print LaTeX tables")
    
    # Dataset selection
    parser.add_argument("--dataset", choices=datasets + ["all"], default="all", 
                        help="Dataset to evaluate (default: all)")
    
    # If no arguments are provided, show help
    if len(sys.argv) == 1:
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
        
        # Create output directory if it doesn't exist
        os.makedirs("../results_align/tables", exist_ok=True)
        
        # Generate combined table
        output_filepath = f"../results_align/tables/combined_table.tex"
        with open(output_filepath, "w") as output_file:
            print(f"Writing combined table to {output_filepath}")
            print_combined_table_latex(results_scores, stats_results, output_file=output_file)
        
        print(f"Combined table written to {output_filepath}")
        
        # If dataset-specific tables are also requested
        if args.dataset != "all":
            dataset = args.dataset
            if dataset in datasets:
                output_filepath = f"../results_align/tables/{dataset}_tables.tex"
                with open(output_filepath, "w") as output_file:
                    print(f"Writing tables for {dataset.upper()} dataset to {output_filepath}")
                    print_table_latex_per_dataset(results_scores, stats_results, dataset=dataset, output_file=output_file)
                print(f"Tables for {dataset.upper()} dataset written to {output_filepath}")


if __name__ == "__main__":
    main()
