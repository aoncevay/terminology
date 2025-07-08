#!/usr/bin/env python3
# Script to analyze the impact of specialized prompts on different domain categories
# Generates comparative group bar plots showing Term Accuracy differences across domain categories

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
import pandas as pd

# Constants
DATASETS = ["irs", "cfpb"]
DIRECTIONS = ["en-xx", "xx-en"]
METRIC = "term-acc"  # Focus only on term accuracy

# Dataset to language mappings
DATASET2LANGS = {
    "irs": ["es", "kr", "ru", "vi", "zh_s", "zh_t", "ht"],
    "cfpb": ["es", "kr", "ru", "vi", "zh_t", "ht"]
}

# Language name mapping for prettier labels
LANG2SHORT = {
    "es": "es",
    "kr": "ko", 
    "ru": "ru",
    "vi": "vi",
    "zh_s": "zh(s)",
    "zh_t": "zh(t)",
    "ht": "ht"
}

# Domain category mappings (from domain_specific_analysis.py)
LABEL2LATEX = {
    "TS": "DS",  # tax specific -> domain specific
    "TC": "DC",  # tax contextual -> domain contextual
    "G": "G",    # general -> general
    "FS": "DS",  # financial specific -> domain specific
    "FC": "DC",  # financial contextual -> domain contextual
}

# Category name mappings for LaTeX
CATEGORYNAME2LATEX = {
    "DS": "Domain-Specific",
    "DC": "Domain-Contextual", 
    "G": "General"
}

# Define models to compare
BASE_MODEL = "LLM_openai_gpt4o"
PROMPT_MODEL = "Task2_LLM_openai_gpt4o"

# Color scheme for domain categories
CATEGORY_COLORS = {
    "DS": "#1f77b4",  # Blue for domain-specific
    "DC": "#ff7f0e",  # Orange for domain-contextual
    "G": "#2ca02c"    # Green for general
}

def load_annotations(dataset):
    """Load term annotations from CSV file"""
    annotation_path = f"../annotations/{dataset}_terminology_labelled_compiled_annotated.csv"
    
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
    annotations = pd.read_csv(annotation_path)
    
    # Create a dictionary mapping English terms to their domain categories
    term_categories = {}
    for _, row in annotations.iterrows():
        if row['in_experiments'] == 1:  # Only include terms used in experiments
            # Map original annotations to standardized labels (DS, DC, G)
            category = LABEL2LATEX.get(row['human_annotation'], row['human_annotation'])
            term_categories[row['full_terms'].lower()] = category
    
    return term_categories

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

def load_term_level_results(models, dataset):
    """Load term-level results for both models"""
    languages = DATASET2LANGS[dataset]
    term_results = {}
    
    for model in models:
        term_results[model] = {}
        
        # Load the values file which contains term-level accuracy
        values_path = f"../results/values_{model}.json"
        if not os.path.exists(values_path):
            print(f"Warning: Results file not found for model {model}: {values_path}")
            continue
            
        with open(values_path, "r", encoding="utf-8") as f:
            values_data = json.load(f)
            
        if dataset not in values_data:
            print(f"Warning: Dataset {dataset} not found in results for model {model}")
            continue
            
        # Process each language and direction
        for lang in languages:
            term_results[model][lang] = {}
            
            for direction in DIRECTIONS:
                if direction == "en-xx":
                    lang_pair = f"en-{lang}"
                else:
                    lang_pair = f"{lang}-en"
                
                # Check if we have data for this language pair
                if lang_pair not in values_data[dataset]:
                    print(f"Warning: No data for {lang_pair} in {dataset} for model {model}")
                    continue
                
                # Get term accuracy binary values (0/1) for each sentence
                term_values = values_data[dataset][lang_pair].get("term_acc", [])
                if not term_values:
                    print(f"Warning: No term accuracy values for {lang_pair} in {dataset} for model {model}")
                    continue
                
                term_results[model][lang][direction] = term_values
    
    return term_results

def categorize_term_results_by_domain(term_results, term_categories, dataset):
    """Categorize term results by domain specificity"""
    categorized_results = {}
    
    for model in term_results:
        categorized_results[model] = {}
        
        for lang in term_results[model]:
            categorized_results[model][lang] = {}
            
            for direction in term_results[model][lang]:
                categorized_results[model][lang][direction] = {"DS": [], "DC": [], "G": []}
                
                # Get the term accuracy binary values
                term_values = term_results[model][lang][direction]
                
                # Skip if no values
                if not term_values:
                    continue
                
                # Load the specific output file to get terms
                if direction == "en-xx":
                    lang_pair = f"en-{lang}"
                else:
                    lang_pair = f"{lang}-en"
                    
                output_path = f"../output/{model}.{dataset}.{lang_pair}.json"
                output_path_v2 = f"../output/{model}_{dataset}_{lang_pair}.json"
                
                if os.path.exists(output_path):
                    with open(output_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                elif os.path.exists(output_path_v2):
                    with open(output_path_v2, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    print(f"Warning: No output file found for {model}.{dataset}.{lang_pair}")
                    continue
                
                # Process each term pair
                for i, term_pair in enumerate(data.get("term_pairs", [])):
                    if i >= len(term_values):
                        break
                        
                    # Extract term and get category
                    if direction == "en-xx":
                        # For en-xx, English is the source (key in dictionary)
                        for en_term in term_pair.keys():
                            category = term_categories.get(en_term.lower())
                            if category:
                                for val in term_values[i]:  # term_values[i] is a list of binary values for this sentence
                                    categorized_results[model][lang][direction][category].append(val)
                            break  # Just use the first term
                    else:
                        # For xx-en, English is the target (value in dictionary)
                        for _, en_term in term_pair.items():
                            category = term_categories.get(en_term.lower())
                            if category:
                                for val in term_values[i]:  # term_values[i] is a list of binary values for this sentence
                                    categorized_results[model][lang][direction][category].append(val)
                            break  # Just use the first term
    
    return categorized_results

def calculate_domain_differences_with_significance(categorized_results, dataset, direction):
    """
    Calculate term accuracy differences by domain category between prompt and base models
    """
    categories = ["DS", "DC", "G"]
    differences = {cat: [] for cat in categories}
    significant = {cat: False for cat in categories}
    
    print(f"\n=== Calculating domain differences for {dataset} {direction} ===")
    
    # Collect all values for each category across languages
    base_values = {cat: [] for cat in categories}
    prompt_values = {cat: [] for cat in categories}
    
    languages = DATASET2LANGS[dataset]
    
    for lang in languages:
        # Skip if data is missing for either model
        if (lang not in categorized_results[BASE_MODEL] or 
            lang not in categorized_results[PROMPT_MODEL] or
            direction not in categorized_results[BASE_MODEL][lang] or
            direction not in categorized_results[PROMPT_MODEL][lang]):
            continue
        
        # Collect values for each category
        for category in categories:
            base_cat_values = categorized_results[BASE_MODEL][lang][direction].get(category, [])
            prompt_cat_values = categorized_results[PROMPT_MODEL][lang][direction].get(category, [])
            
            base_values[category].extend(base_cat_values)
            prompt_values[category].extend(prompt_cat_values)
    
    # Calculate differences and significance for each category
    for category in categories:
        if not base_values[category] or not prompt_values[category]:
            print(f"Warning: No data for category {category}")
            continue
            
        # Calculate mean accuracy for each model
        base_acc = sum(base_values[category]) / len(base_values[category])
        prompt_acc = sum(prompt_values[category]) / len(prompt_values[category])
        
        # Calculate difference (prompt - base)
        difference = prompt_acc - base_acc
        differences[category] = difference
        
        print(f"{category}: Base={base_acc:.4f}, Prompt={prompt_acc:.4f}, Diff={difference:.4f}")
        
        # Perform statistical significance test
        if len(base_values[category]) >= 5 and len(prompt_values[category]) >= 5:
            try:
                u_stat, p_value = stats.mannwhitneyu(base_values[category], prompt_values[category], 
                                                   alternative='two-sided')
                significant[category] = p_value < 0.05
                print(f"  Mann-Whitney U test: U={u_stat:.2f}, p={p_value:.4f}, Significant: {significant[category]}")
            except ValueError as e:
                print(f"  Mann-Whitney U test failed: {e}")
                # Fall back to simple threshold
                significant[category] = abs(difference) >= 0.10
        else:
            # Use simple threshold for term accuracy (10%)
            significant[category] = abs(difference) >= 0.10
            print(f"  Using simple threshold: |{difference:.4f}| >= 0.10 -> {significant[category]}")
    
    return differences, significant

def create_domain_difference_plot(differences, significant, dataset, direction, output_dir="../figs/domain_prompt_analysis"):
    """Create a group bar plot showing term accuracy differences by domain category"""
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 2.5))
    
    # Prepare data
    categories = ["DS", "DC", "G"]
    category_labels = [CATEGORYNAME2LATEX[cat] for cat in categories]
    values = [differences.get(cat, 0) for cat in categories]
    colors = [CATEGORY_COLORS[cat] for cat in categories]
    
    # Create bars
    bars = ax.bar(
        range(len(categories)),
        values,
        color=colors,
        edgecolor='black',
        linewidth=0.5,
        width=0.6,
        alpha=0.8
    )
    
    # Add significance markers (asterisks)
    print(f"\nAdding significance markers for {dataset} {direction}:")
    for i, category in enumerate(categories):
        if significant.get(category, False):
            value = differences.get(category, 0)
            # Position the marker above or below the bar
            if value >= 0:
                y_pos = value + 0.01  # Slightly above positive bars
            else:
                y_pos = value - 0.01  # Slightly below negative bars
            
            # Use black for all stars for visibility
            ax.text(i, y_pos, '*', ha='center', va='center', fontsize=14, 
                   fontweight='bold', color='black')
            print(f"    Added * for {category} at position ({i}, {y_pos})")
    
    # Add category labels
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(category_labels, rotation=0)
    
    # Add zero reference line
    ax.axhline(y=0, color='green', linestyle='-', linewidth=1)
    
    # Set y-axis label
    ax.set_ylabel("Term Acc diff")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure - PDF only
    filename = f"domain_gpt4o_{dataset}_{direction}_term_acc_diff.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    
    print(f"Saved plot to {filepath}")
    
    # Close figure to free memory
    plt.close()
    
    return filepath

def generate_latex_code(filepaths):
    """Generate LaTeX code to include all domain category figures"""
    latex_code = "% LaTeX code to include domain prompt analysis figures\n\n"
    
    # Group filepaths by dataset
    by_dataset = {}
    
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        
        # Expected format: domain_gpt4o_dataset_direction_term_acc_diff.pdf
        if not filename.startswith("domain_gpt4o_") or not filename.endswith("_term_acc_diff.pdf"):
            print(f"Warning: Unexpected filename format: {filename}")
            continue
        
        # Extract dataset from filename
        parts = filename.split('_')
        if len(parts) < 4:
            print(f"Warning: Filename doesn't have enough parts: {filename}")
            continue
            
        dataset = parts[2]  # domain_gpt4o_DATASET_direction_...
        direction = parts[3]
        
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append((direction, filepath))
    
    # Generate LaTeX code for each dataset
    for dataset, direction_paths in by_dataset.items():
        latex_code += f"% {dataset.upper()} Term Accuracy by Domain Category comparison\n"
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
            latex_code += f"        \\label{{fig:domain_prompt_diff_{dataset}_{subfig_label}}}\n"
            latex_code += f"    \\end{{subfigure}}\n"
            
            # Add space between subfigures if this is the first one
            if i == 0 and len(sorted_paths) > 1:
                latex_code += f"    \\hfill\n"
        
        # Caption and label
        latex_code += f"    \\caption{{Difference in Term Accuracy scores by domain category between GPT4o with specialized prompt and standard GPT4o for {dataset.upper()} dataset. Asterisks (*) indicate statistically significant differences.}}\n"
        latex_code += f"    \\label{{fig:domain_prompt_diff_{dataset}}}\n"
        latex_code += "\\end{figure}\n\n"
    
    # Save LaTeX code to file
    latex_path = "../figs/domain_prompt_analysis/domain_prompt_figures.tex"
    os.makedirs(os.path.dirname(latex_path), exist_ok=True)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    
    print(f"LaTeX code saved to {latex_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze domain-specific prompt impact on GPT4o performance")
    parser.add_argument("--output_dir", default="../figs/domain_prompt_analysis", 
                        help="Directory to save output figures")
    parser.add_argument("--latex", action="store_true", 
                        help="Generate LaTeX code for including figures")
    parser.add_argument("--dataset", choices=["irs", "cfpb"], default="irs",
                        help="Dataset to analyze (default: irs)")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load domain annotations
    print(f"Loading domain annotations for {args.dataset}...")
    term_categories = load_annotations(args.dataset)
    print(f"Loaded {len(term_categories)} annotated terms with domain categories")
    
    # Load results
    print("Loading evaluation results...")
    results = load_results()
    
    # Check if we have data for both models
    if not results[BASE_MODEL] or not results[PROMPT_MODEL]:
        print("Error: Missing data for one or both models")
        return
    
    # Load term-level results for domain categorization
    print("Loading term-level results...")
    models = [BASE_MODEL, PROMPT_MODEL]
    term_results = load_term_level_results(models, args.dataset)
    
    # Categorize results by domain
    print("Categorizing results by domain specificity...")
    categorized_results = categorize_term_results_by_domain(term_results, term_categories, args.dataset)
    
    # Track all generated filepaths
    all_filepaths = []
    
    # Calculate differences and create plots for each direction
    for direction in DIRECTIONS:
        print(f"\nProcessing {direction} direction...")
        
        # Calculate differences and significance
        differences, significant = calculate_domain_differences_with_significance(
            categorized_results, args.dataset, direction
        )
        
        # Skip if no data
        if not any(differences.values()):
            print(f"No data for {args.dataset} {direction}")
            continue
        
        # Create plot
        filepath = create_domain_difference_plot(differences, significant, args.dataset, direction, 
                                               output_dir=args.output_dir)
        all_filepaths.append(filepath)
    
    # Generate LaTeX code if requested
    if args.latex and all_filepaths:
        generate_latex_code(all_filepaths)
    
    print(f"\nGenerated {len(all_filepaths)} plots for domain category analysis")

if __name__ == "__main__":
    main()
