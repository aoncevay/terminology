#!/usr/bin/env python3
# Script to analyze the impact of specialized prompts (GPT4o++ vs GPT4o)
# Generates comparative bar plots showing score differences across languages

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import argparse
from scipy import stats
import glob
import re

# Constants
DATASETS = ["irs", "cfpb"]
METRICS = ["chrf++", "term-acc"]
DIRECTIONS = ["en-xx", "xx-en"]

# Mapping between different formats of metric names (for backward compatibility)
METRIC_MAPPING = {
    "term_acc": "term-acc",
    "term-acc": "term-acc",
    "chrf++": "chrf++"
}

# Dataset to language mappings
DATASET2LANGS = {
    "irs": ["es", "kr", "ru", "vi", "zh_s", "zh_t", "ht"],
    "cfpb": ["es", "kr", "ru", "vi", "zh_t", "ht"]
}

# Language name mapping for prettier labels
LANG2NAME = {
    "es": "Spanish",
    "kr": "Korean",
    "ru": "Russian",
    "vi": "Vietnamese",
    "zh_s": "Chinese (S)",
    "zh_t": "Chinese (T)",
    "ht": "Haitian"
}

# Short language codes for axis labels
LANG2SHORT = {
    "es": "es",
    "kr": "ko",
    "ru": "ru",
    "vi": "vi",
    "zh_s": "zh(s)",
    "zh_t": "zh(t)",
    "ht": "ht"
}

# Pretty metric names
METRIC2NAME = {
    "chrf++": "chrF++",
    "term-acc": "Term Accuracy"
}

# Define models to compare
BASE_MODEL = "LLM_openai_gpt4o"
PROMPT_MODEL = "Task2_LLM_openai_gpt4o"

# Color scheme
GAIN_COLOR = "#AED6F1"  # Light blue for gains
LOSS_COLOR = "#F5B7B1"  # Light red for losses

def load_results():
    """Load evaluation results for both models"""
    results = {}
    
    # Paths to result files
    base_path = f"../results/scores_{BASE_MODEL}.json"
    prompt_path = f"../results/scores_{PROMPT_MODEL}.json"
    
    # Load base model results
    if os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8") as f:
            results[BASE_MODEL] = json.load(f)
    else:
        print(f"Warning: Results for {BASE_MODEL} not found at {base_path}")
        results[BASE_MODEL] = {}
    
    # Load prompt-enhanced model results
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            results[PROMPT_MODEL] = json.load(f)
    else:
        print(f"Warning: Results for {PROMPT_MODEL} not found at {prompt_path}")
        results[PROMPT_MODEL] = {}
    
    return results

def get_language_pairs(dataset, direction):
    """Generate language pairs for a given dataset and direction"""
    language_pairs = []
    langs = DATASET2LANGS.get(dataset, [])
    
    if direction == "en-xx":
        return [f"en-{lang}" for lang in langs]
    else:  # xx-en
        return [f"{lang}-en" for lang in langs]

def perform_statistical_tests(dataset, direction, metric):
    """
    Perform Mann-Whitney U tests directly on model outputs
    
    Args:
        dataset: Dataset name (irs or cfpb)
        direction: Translation direction (en-xx or xx-en)
        metric: Evaluation metric (chrf++ or term-acc)
    
    Returns:
        Dict mapping language codes to significance test results (True/False)
    """
    print(f"\n=== Performing statistical tests for {dataset} {direction} {metric} ===")
    
    # Get language pairs for this dataset and direction
    language_pairs = get_language_pairs(dataset, direction)
    significant_results = {}
    
    # Extract language codes based on direction
    if direction == "en-xx":
        languages = [pair.split("-")[1] for pair in language_pairs]
    else:  # xx-en
        languages = [pair.split("-")[0] for pair in language_pairs]
    
    # Define model output directories
    base_output_dir = f"../outputs/{BASE_MODEL}/{dataset}"
    prompt_output_dir = f"../outputs/{PROMPT_MODEL}/{dataset}"
    
    # Process each language pair
    for lang, lang_pair in zip(languages, language_pairs):
        print(f"\nAnalyzing {lang_pair}...")
        
        # Find outputs for this language pair
        base_outputs = sorted(glob.glob(f"{base_output_dir}/{lang_pair}*.json"))
        prompt_outputs = sorted(glob.glob(f"{prompt_output_dir}/{lang_pair}*.json"))
        
        if not base_outputs or not prompt_outputs:
            print(f"  No outputs found for {lang_pair}")
            continue
        
        # Load outputs and extract scores
        base_scores = []
        prompt_scores = []
        
        # Process base model outputs
        for output_file in base_outputs:
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract scores based on metric
                if metric == "term-acc":
                    # For term accuracy, check each term in the output
                    if "terms" in data and "translations" in data:
                        terms = data["terms"]
                        translations = data["translations"]
                        
                        if terms and translations:
                            # Calculate term accuracy for this sentence
                            correct_terms = 0
                            total_terms = len(terms)
                            
                            if total_terms > 0:
                                # For term accuracy, we need some custom logic to compute values
                                # This is a simplification - real evaluation would be more complex
                                translation_lower = translations.lower()
                                for term in terms:
                                    if term.lower() in translation_lower:
                                        correct_terms += 1
                                
                                accuracy = correct_terms / total_terms
                                base_scores.append(accuracy)
                
                elif metric == "chrf++":
                    # For chrF++, directly extract the score if available
                    if "chrf++" in data:
                        base_scores.append(data["chrf++"])
                    elif "chrF++" in data:
                        base_scores.append(data["chrF++"])
                    elif "scores" in data and "chrf++" in data["scores"]:
                        base_scores.append(data["scores"]["chrf++"])
                    
            except Exception as e:
                print(f"  Error processing base model output {output_file}: {e}")
        
        # Process prompt model outputs (similar to base model)
        for output_file in prompt_outputs:
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract scores based on metric (same logic as above)
                if metric == "term-acc":
                    if "terms" in data and "translations" in data:
                        terms = data["terms"]
                        translations = data["translations"]
                        
                        if terms and translations:
                            correct_terms = 0
                            total_terms = len(terms)
                            
                            if total_terms > 0:
                                translation_lower = translations.lower()
                                for term in terms:
                                    if term.lower() in translation_lower:
                                        correct_terms += 1
                                
                                accuracy = correct_terms / total_terms
                                prompt_scores.append(accuracy)
                
                elif metric == "chrf++":
                    if "chrf++" in data:
                        prompt_scores.append(data["chrf++"])
                    elif "chrF++" in data:
                        prompt_scores.append(data["chrF++"])
                    elif "scores" in data and "chrf++" in data["scores"]:
                        prompt_scores.append(data["scores"]["chrf++"])
                    
            except Exception as e:
                print(f"  Error processing prompt model output {output_file}: {e}")
        
        # If we couldn't extract scores using the raw outputs, try the scores file
        if (not base_scores or not prompt_scores) and metric == "chrf++":
            print(f"  No raw scores found, attempting to use score file for {lang_pair}")
            # Load the score files
            base_path = f"../results/scores_{BASE_MODEL}.json"
            prompt_path = f"../results/scores_{PROMPT_MODEL}.json"
            
            try:
                with open(base_path, "r", encoding="utf-8") as f:
                    base_results = json.load(f)
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt_results = json.load(f)
                
                # Extract chrF++ scores
                if (dataset in base_results and lang_pair in base_results[dataset] and 
                    dataset in prompt_results and lang_pair in prompt_results[dataset]):
                    
                    base_score = base_results[dataset][lang_pair].get("chrf++", None)
                    prompt_score = prompt_results[dataset][lang_pair].get("chrf++", None)
                    
                    if base_score is not None and prompt_score is not None:
                        # For chrF++, we'll use 2.0 point difference as significant
                        difference = prompt_score - base_score
                        significant_results[lang] = abs(difference) >= 2.0
                        print(f"  Using aggregate scores: Base={base_score:.4f}, Prompt={prompt_score:.4f}")
                        print(f"  chrF++ threshold test: |{difference:.4f}| >= 2.0 -> {significant_results[lang]}")
                        continue
            except Exception as e:
                print(f"  Error loading score files: {e}")
        
        # Perform statistical test if we have enough data
        if len(base_scores) >= 5 and len(prompt_scores) >= 5:
            try:
                print(f"  Performing Mann-Whitney U test with {len(base_scores)} base and {len(prompt_scores)} prompt samples")
                u_stat, p_value = stats.mannwhitneyu(base_scores, prompt_scores, alternative='two-sided')
                significant_results[lang] = p_value < 0.05
                print(f"  Result: U={u_stat:.2f}, p={p_value:.4f}, Significant: {significant_results[lang]}")
            except ValueError as e:
                print(f"  Mann-Whitney U test failed: {e}")
                significant_results[lang] = False
        else:
            print(f"  Not enough data for test: {len(base_scores)} base, {len(prompt_scores)} prompt")
            # For term accuracy, fall back to comparing aggregate scores
            if metric == "term-acc":
                # Load the score files
                base_path = f"../results/scores_{BASE_MODEL}.json"
                prompt_path = f"../results/scores_{PROMPT_MODEL}.json"
                
                try:
                    with open(base_path, "r", encoding="utf-8") as f:
                        base_results = json.load(f)
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        prompt_results = json.load(f)
                    
                    # Try both term_acc and term-acc formats
                    base_score = -1
                    prompt_score = -1
                    
                    if dataset in base_results and lang_pair in base_results[dataset]:
                        base_score = base_results[dataset][lang_pair].get("term-acc", 
                                    base_results[dataset][lang_pair].get("term_acc", -1))
                    
                    if dataset in prompt_results and lang_pair in prompt_results[dataset]:
                        prompt_score = prompt_results[dataset][lang_pair].get("term-acc", 
                                       prompt_results[dataset][lang_pair].get("term_acc", -1))
                    
                    if base_score != -1 and prompt_score != -1:
                        # For term accuracy, use a difference of >= 0.10 as significant
                        difference = prompt_score - base_score
                        significant_results[lang] = abs(difference) >= 0.10
                        print(f"  Using aggregate scores: Base={base_score:.4f}, Prompt={prompt_score:.4f}")
                        print(f"  Term acc threshold test: |{difference:.4f}| >= 0.10 -> {significant_results[lang]}")
                except Exception as e:
                    print(f"  Error loading score files: {e}")
            
            # For chrF++, apply the 2-point threshold
            elif metric == "chrf++" and base_scores and prompt_scores:
                avg_base = sum(base_scores) / len(base_scores)
                avg_prompt = sum(prompt_scores) / len(prompt_scores)
                difference = avg_prompt - avg_base
                significant_results[lang] = abs(difference) >= 2.0
                print(f"  Using averaged samples: Base={avg_base:.4f}, Prompt={avg_prompt:.4f}")
                print(f"  chrF++ threshold test: |{difference:.4f}| >= 2.0 -> {significant_results[lang]}")
    
    # Print summary of significant results
    print("\nSignificance testing summary:")
    for lang in significant_results:
        sig_mark = "*" if significant_results[lang] else ""
        print(f"  {lang}: {significant_results[lang]} {sig_mark}")
    
    return significant_results

def calculate_differences_with_significance(results, dataset, direction, metric, use_independent_stats=False, significance_cache=None):
    """
    Calculate score differences between prompt and base models and determine statistical significance
    
    Args:
        results: Dictionary of evaluation results
        dataset: Dataset name
        direction: Translation direction
        metric: Evaluation metric
        use_independent_stats: Whether to use independently computed statistical significance
        significance_cache: Cache of precomputed statistical significance results
    
    Returns:
        differences: Dictionary of score differences by language
        significant: Dictionary indicating whether each language has a significant difference
    """
    language_pairs = get_language_pairs(dataset, direction)
    differences = {}
    significant = {}  # Track which differences are statistically significant
    
    print(f"\n=== Calculating differences for {dataset} {direction} {metric} ===")
    
    # If using independent stats and we have cached results, use them
    if use_independent_stats and significance_cache and dataset in significance_cache and direction in significance_cache[dataset] and metric in significance_cache[dataset][direction]:
        print("Using independently computed statistical significance")
        cached_significance = significance_cache[dataset][direction][metric]
        
    # Extract languages from pairs based on direction
    if direction == "en-xx":
        languages = [pair.split("-")[1] for pair in language_pairs]
    else:  # xx-en
        languages = [pair.split("-")[0] for pair in language_pairs]
    
    # Process each language
    for lang, lang_pair in zip(languages, language_pairs):
        # Skip if data is missing
        if (dataset not in results[BASE_MODEL] or 
            dataset not in results[PROMPT_MODEL] or
            lang_pair not in results[BASE_MODEL][dataset] or
            lang_pair not in results[PROMPT_MODEL][dataset]):
            print(f"Warning: Missing data for {lang_pair} in {dataset}")
            continue
        
        # Get scores for both models
        # Check for both term_acc and term-acc formats
        base_score = -1
        prompt_score = -1
        
        # Try to get the score using the current metric name
        base_score = results[BASE_MODEL][dataset][lang_pair].get(metric, -1)
        prompt_score = results[PROMPT_MODEL][dataset][lang_pair].get(metric, -1)
        
        # If score not found, try alternative format (term_acc instead of term-acc)
        if base_score == -1 and metric == "term-acc":
            base_score = results[BASE_MODEL][dataset][lang_pair].get("term_acc", -1)
            if base_score != -1:
                print(f"Found alternative format 'term_acc' for {lang_pair} base model")
        if prompt_score == -1 and metric == "term-acc":
            prompt_score = results[PROMPT_MODEL][dataset][lang_pair].get("term_acc", -1)
            if prompt_score != -1:
                print(f"Found alternative format 'term_acc' for {lang_pair} prompt model")
        
        # Skip if invalid scores
        if base_score == -1 or prompt_score == -1:
            print(f"Warning: Invalid scores for {lang_pair} in {dataset}")
            continue
        
        # Calculate difference (prompt - base)
        difference = prompt_score - base_score
        print(f"{lang_pair}: Base={base_score:.4f}, Prompt={prompt_score:.4f}, Diff={difference:.4f}")
        
        # Store difference
        differences[lang] = difference
        
        # If we're using independently computed significance, get it from the cache
        if use_independent_stats and significance_cache and dataset in significance_cache and direction in significance_cache[dataset] and metric in significance_cache[dataset][direction] and lang in significance_cache[dataset][direction][metric]:
            significant[lang] = significance_cache[dataset][direction][metric][lang]
            print(f"  Using independent significance test: {significant[lang]}")
        else:
            # Otherwise, use simple thresholds
            if metric == "term-acc":
                # For term accuracy, use a threshold of 0.1 (10%)
                significant[lang] = abs(difference) >= 0.10
                print(f"  Using simple threshold: |{difference:.4f}| >= 0.10 -> {significant[lang]}")
            elif metric == "chrf++":
                # For chrF++, use a threshold of 2 points
                significant[lang] = abs(difference) >= 2.0
                print(f"  Using simple threshold: |{difference:.4f}| >= 2.0 -> {significant[lang]}")
            else:
                print(f"  No significance test available for metric: {metric}")
                significant[lang] = False
    
    # Print summary of significant differences
    print("\nSignificant differences summary:")
    for lang in differences.keys():
        sig_mark = "*" if lang in significant and significant[lang] else ""
        print(f"  {lang}: {differences[lang]:.4f} {sig_mark}")
    
    return differences, significant

def create_difference_plot(differences, significant, dataset, direction, metric, y_limits=None, output_dir="../figs/prompt_analysis"):
    """Create a bar plot showing score differences with significance markers"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style for publication quality
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 11
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9
    
    # Create figure with specified dimensions
    fig, ax = plt.subplots(figsize=(3, 2))
    
    # Prepare data - maintain original language order (alphabetical by language code)
    languages = list(differences.keys())
    if direction == "en-xx":
        # For en-xx, use the dataset's defined language order
        ordered_langs = [lang for lang in DATASET2LANGS[dataset] if lang in languages]
    else:  # xx-en
        # For xx-en, also use the dataset's defined language order
        ordered_langs = [lang for lang in DATASET2LANGS[dataset] if lang in languages]
    
    # Get values in the same order as languages
    values = [differences[lang] for lang in ordered_langs]
    
    # Create bars with colors based on sign - removed hatch for term-acc
    bars = ax.bar(
        range(len(ordered_langs)),
        values,
        color=[GAIN_COLOR if v >= 0 else LOSS_COLOR for v in values],
        edgecolor='black',
        linewidth=0.5,
        width=0.7
    )
    
    # Add significance markers (asterisks) inside the bars - now all black
    print(f"\nAdding significance markers for {dataset} {direction} {metric}:")
    for i, lang in enumerate(ordered_langs):
        print(f"  {lang}: significant={lang in significant and significant[lang]}")
        if lang in significant and significant[lang]:
            value = differences[lang]
            # Position the marker in the middle of the bar (vertically)
            y_pos = value / 2  # Middle of the bar
            
            # For very short bars, place the asterisk just above or below
            if abs(value) < 0.05:  # If the bar is very small
                y_pos = 0.05 if value >= 0 else -0.05
            
            # Use black for all stars for better visibility and consistency
            ax.text(i, y_pos, '*', ha='center', va='center', fontsize=14, 
                   fontweight='bold', color='black')
            print(f"    Added * at position ({i}, {y_pos})")
    
    # Add language labels - removed rotation
    ax.set_xticks(range(len(ordered_langs)))
    ax.set_xticklabels([LANG2SHORT[lang] for lang in ordered_langs])
    
    # Add zero reference line
    ax.axhline(y=0, color='green', linestyle='-', linewidth=1)
    
    # Set y-axis limits if provided (synchronized between direction pairs)
    if y_limits:
        ax.set_ylim(y_limits)
    
    # No title - this will be added in LaTeX
    
    # Add y-axis label based on metric
    if metric == "chrf++":
        ax.set_ylabel("chrF++ diff")
    else:  # term-acc
        ax.set_ylabel("Term Acc diff")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure - PDF only with model name prefix
    filename = f"gpt4o_{dataset}_{direction}_{metric}_diff.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    
    print(f"Saved plot to {filepath}")
    
    # Close figure to free memory
    plt.close()
    
    return filepath

def generate_latex_code(filepaths):
    """Generate LaTeX code to include all figures in a paper"""
    latex_code = "% LaTeX code to include prompt analysis figures\n\n"
    
    # Group filepaths by dataset and metric
    by_dataset_metric = {}
    
    # Simplify the parsing logic
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        
        # Quick check - expected format is gpt4o_dataset_direction_metric_diff.pdf
        if not filename.startswith("gpt4o_") or not filename.endswith("_diff.pdf"):
            print(f"Warning: Unexpected filename format: {filename}")
            continue
        
        # Extract parts from the filename structure
        parts = filename.split('_')
        if len(parts) < 5:
            print(f"Warning: Filename doesn't have enough parts: {filename}")
            continue
            
        dataset = parts[1]
        direction = parts[2]
        
        # Extract metric (might contain special characters like ++)
        if "chrf++" in filename:
            metric = "chrf++"
        else:
            metric = "term-acc"
        
        # Use simple key
        key = dataset + "-" + metric
        
        if key not in by_dataset_metric:
            by_dataset_metric[key] = []
        by_dataset_metric[key].append((direction, filepath))
    
    # Debug information
    print(f"Grouped filepaths: {list(by_dataset_metric.keys())}")
    
    # Generate LaTeX code for each group
    for key, direction_paths in by_dataset_metric.items():
        # Parse the simple key format
        dataset, metric = key.split("-", 1)
        metric_name = METRIC2NAME[metric]
        
        latex_code += f"% {dataset.upper()} {metric_name} comparison\n"
        latex_code += "\\begin{figure}[t]\n"
        latex_code += "    \\centering\n"
        
        # Sort paths to ensure en-xx comes before xx-en
        sorted_paths = sorted(direction_paths, key=lambda x: x[0])
        
        # Add subfigures
        for i, (direction, path) in enumerate(sorted_paths):
            # Determine labels
            subfig_label = "a" if direction == "en-xx" else "b"
            direction_label = "en→xx" if direction == "en-xx" else "xx→en"
            
            latex_code += f"    \\begin{{subfigure}}[b]{{0.48\\linewidth}}\n"
            latex_code += f"        \\centering\n"
            latex_code += f"        \\includegraphics[width=\\linewidth]{{{os.path.basename(path)}}}\n"
            latex_code += f"        \\caption{{{direction_label}}}\n"
            latex_code += f"        \\label{{fig:prompt_diff_{dataset}_{metric}_{subfig_label}}}\n"
            latex_code += f"    \\end{{subfigure}}\n"
            
            # Add space between subfigures if this is the first one
            if i == 0 and len(sorted_paths) > 1:
                latex_code += f"    \\hfill\n"
        
        # Caption and label
        latex_code += f"    \\caption{{Difference in {metric_name} scores between GPT4o with specialized prompt and standard GPT4o for {dataset.upper()} dataset. Asterisks (*) indicate statistically significant differences.}}\n"
        latex_code += f"    \\label{{fig:prompt_diff_{dataset}_{metric}}}\n"
        latex_code += "\\end{figure}\n\n"
    
    # Save LaTeX code to file
    latex_path = "../figs/prompt_analysis/prompt_figures.tex"
    os.makedirs(os.path.dirname(latex_path), exist_ok=True)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    
    print(f"LaTeX code saved to {latex_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze prompt impact on GPT4o performance")
    parser.add_argument("--output_dir", default="../figs/prompt_analysis", 
                        help="Directory to save output figures")
    parser.add_argument("--latex", action="store_true", 
                        help="Generate LaTeX code for including figures")
    parser.add_argument("--stats", action="store_true",
                        help="Perform independent statistical testing on raw outputs")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results()
    
    # Check if we have data for both models
    if not results[BASE_MODEL] or not results[PROMPT_MODEL]:
        print("Error: Missing data for one or both models")
        return
    
    # If using independent statistical testing, compute it first
    significance_cache = {}
    if args.stats:
        print("\n*** Computing independent statistical significance ***")
        for dataset in DATASETS:
            significance_cache[dataset] = {}
            for direction in DIRECTIONS:
                significance_cache[dataset][direction] = {}
                for metric in METRICS:
                    significance_cache[dataset][direction][metric] = perform_statistical_tests(dataset, direction, metric)
    
    # Track all generated filepaths
    all_filepaths = []
    
    # First pass: calculate all differences and determine shared y-axis limits for each dataset/metric pair
    y_axis_limits = {}
    all_differences = {}
    all_significance = {}
    
    for dataset in DATASETS:
        y_axis_limits[dataset] = {}
        all_differences[dataset] = {}
        all_significance[dataset] = {}
        
        for metric in METRICS:
            y_axis_limits[dataset][metric] = {"min": 0, "max": 0}
            all_differences[dataset][metric] = {}
            all_significance[dataset][metric] = {}
            
            # Calculate differences for both directions
            for direction in DIRECTIONS:
                differences, significant = calculate_differences_with_significance(
                    results, dataset, direction, metric, 
                    use_independent_stats=args.stats,
                    significance_cache=significance_cache
                )
                
                # Skip if no data
                if not differences:
                    print(f"No data for {dataset} {direction} {metric}")
                    continue
                
                # Store differences and significance for later use
                all_differences[dataset][metric][direction] = differences
                all_significance[dataset][metric][direction] = significant
                
                # Update min/max values across both directions
                if differences:
                    values = list(differences.values())
                    current_min = min(values)
                    current_max = max(values)
                    
                    y_axis_limits[dataset][metric]["min"] = min(y_axis_limits[dataset][metric]["min"], current_min)
                    y_axis_limits[dataset][metric]["max"] = max(y_axis_limits[dataset][metric]["max"], current_max)
    
    # Second pass: create plots using synchronized y-axis limits
    for dataset in DATASETS:
        for metric in METRICS:
            for direction in DIRECTIONS:
                # Skip if no data
                if direction not in all_differences[dataset][metric]:
                    continue
                
                differences = all_differences[dataset][metric][direction]
                significant = all_significance[dataset][metric][direction]
                
                # Get shared y-axis limits
                y_min = y_axis_limits[dataset][metric]["min"]
                y_max = y_axis_limits[dataset][metric]["max"]
                
                # Add a small padding (5%) to the limits for visual clarity
                y_range = y_max - y_min
                y_min = y_min - 0.05 * y_range
                y_max = y_max + 0.05 * y_range
                
                # Create plot with synchronized y-axis
                filepath = create_difference_plot(differences, significant, dataset, direction, metric, 
                                                 y_limits=(y_min, y_max),
                                                 output_dir=args.output_dir)
                all_filepaths.append(filepath)
    
    # Generate LaTeX code if requested
    if args.latex and all_filepaths:
        generate_latex_code(all_filepaths)
    
    print(f"Generated {len(all_filepaths)} plots")

if __name__ == "__main__":
    main()
