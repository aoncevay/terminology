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
from scipy import stats  # Add import for Mann-Whitney U test

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

def calculate_differences_with_significance(results, dataset, direction, metric):
    """
    Calculate score differences between prompt and base models and determine statistical significance
    
    Returns:
        differences: Dictionary of score differences by language
        significant: Dictionary indicating whether each language has a significant difference
    """
    language_pairs = get_language_pairs(dataset, direction)
    differences = {}
    significant = {}  # Track which differences are statistically significant
    
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
        if prompt_score == -1 and metric == "term-acc":
            prompt_score = results[PROMPT_MODEL][dataset][lang_pair].get("term_acc", -1)
        
        # Skip if invalid scores
        if base_score == -1 or prompt_score == -1:
            print(f"Warning: Invalid scores for {lang_pair} in {dataset}")
            continue
        
        # Calculate difference (prompt - base)
        difference = prompt_score - base_score
        
        # Store difference
        differences[lang] = difference
        
        # Determine statistical significance
        if metric == "term-acc" and ("term_acc_values" in results[BASE_MODEL][dataset][lang_pair] or "term-acc_values" in results[BASE_MODEL][dataset][lang_pair]) and ("term_acc_values" in results[PROMPT_MODEL][dataset][lang_pair] or "term-acc_values" in results[PROMPT_MODEL][dataset][lang_pair]):
            # Get term accuracy values (these are binary success/failure values per term)
            # Try both formats of the key (term_acc_values or term-acc_values)
            base_values = results[BASE_MODEL][dataset][lang_pair].get("term_acc_values", results[BASE_MODEL][dataset][lang_pair].get("term-acc_values", []))
            prompt_values = results[PROMPT_MODEL][dataset][lang_pair].get("term_acc_values", results[PROMPT_MODEL][dataset][lang_pair].get("term-acc_values", []))
            
            # Convert to per-sentence accuracy scores for Mann-Whitney U test
            base_per_sentence = []
            prompt_per_sentence = []
            
            for base_sent, prompt_sent in zip(base_values, prompt_values):
                if base_sent and prompt_sent:  # Skip empty sentences
                    base_acc = sum(base_sent) / len(base_sent) if len(base_sent) > 0 else 0
                    prompt_acc = sum(prompt_sent) / len(prompt_sent) if len(prompt_sent) > 0 else 0
                    
                    base_per_sentence.append(base_acc)
                    prompt_per_sentence.append(prompt_acc)
            
            # Perform statistical test if we have enough data
            if len(base_per_sentence) >= 5 and len(prompt_per_sentence) >= 5:
                try:
                    u_stat, p_value = stats.mannwhitneyu(base_per_sentence, prompt_per_sentence, alternative='two-sided')
                    significant[lang] = p_value < 0.05
                except ValueError:
                    # If test cannot be performed, mark as not significant
                    significant[lang] = False
            else:
                significant[lang] = False
                
        elif metric == "chrf++":
            # For chrF++, use a threshold of 2 points to determine significance
            significant[lang] = abs(difference) >= 2.0
        else:
            significant[lang] = False
    
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
    for i, lang in enumerate(ordered_langs):
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
                differences, significant = calculate_differences_with_significance(results, dataset, direction, metric)
                
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
