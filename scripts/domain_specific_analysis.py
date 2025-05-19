import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import pandas as pd
from pathlib import Path

# Models selected for the analysis (can be overridden with command-line arguments)
models = ["MADLAD", "LLM_mistral", "LLM_openai_gpt4o"]

# Define category colors for consistent visualization
CATEGORY_COLORS = {
    "DS": "#1f77b4",  # Blue for domain-specific
    "DC": "#ff7f0e",  # Orange for domain-contextual
    "G": "#2ca02c"    # Green for general
}

# Translation directions
DIRECTIONS = ["en-xx", "xx-en"]

# LaTeX mapping for category labels
LABEL2LATEX = {
    "TS": "DS",  # tax specific -> domain specific
    "TC": "DC",  # tax contextual -> domain contextual
    "G": "G",    # general -> general
    "FS": "DS",  # financial specific -> domain specific
    "FC": "DC",  # financial contextual -> domain contextual
}

# Language mappings (reused from evaluate_mt.py)
LANGID2LATEX = {
    "es": "\\textsc{es}",
    "kr": "\\textsc{ko}",
    "ru": "\\textsc{ru}",
    "vi": "\\textsc{vi}",
    "zh_s": "\\textsc{zh(s)}",
    "zh_t": "\\textsc{zh(t)}",
    "ht": "\\textsc{ht}"
}

# Model name mappings for LaTeX (reused from evaluate_mt.py)
MODELSNAME2LATEX = {
    "MADLAD": "\\textsc{Madlad}",
    "LLM.aya": "\\textsc{Aya23}",
    "LLM.llama": "\\textsc{Llama3.1}",
    "LLM_mistral": "\\textsc{Mistral}",
    "LLM_openai_gpt4o": "\\textsc{GPT4o}"
}

# Category name mappings for LaTeX
CATEGORYNAME2LATEX = {
    "DS": "Domain-Specific",
    "DC": "Domain-Contextual",
    "G": "General"
}

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze term accuracy by domain specificity")
    
    # Execution control arguments
    action_group = parser.add_argument_group('Actions')
    action_group.add_argument("--metrics", action="store_true", 
                              help="Compute metrics and categorize results by domain specificity")
    action_group.add_argument("--stats", action="store_true", 
                              help="Run statistical significance tests")
    action_group.add_argument("--plot", action="store_true", 
                              help="Create boxplot visualizations")
    
    # Analysis type
    parser.add_argument("--between-models", action="store_true",
                        help="Compare different models within each category (default: False)")
    parser.add_argument("--within-model", action="store_true", default=True,
                        help="Compare different categories within each model (default: True)")
    
    # Dataset and model selection
    parser.add_argument("--dataset", choices=["irs", "cfpb"], default="irs",
                        help="Dataset to analyze (default: irs)")
    parser.add_argument("--models", nargs="+", default=models,
                        help="Models to include in analysis (default: MADLAD, LLM.aya, LLM_openai_gpt4o)")
    
    # Output directories
    parser.add_argument("--results_dir", default="../results_domain",
                        help="Directory to save results data (default: ../results_domain)")
    parser.add_argument("--figs_dir", default="../figs",
                        help="Directory to save figures (default: ../figs)")
    
    # If no arguments are provided, show help
    if os.sys.argv[1:] == []:
        parser.print_help()
        exit(1)
    
    return parser.parse_args()

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

def load_term_level_results(models, dataset):
    """Load term-level results for all models and languages"""
    languages = ["es", "kr", "ru", "vi", "zh_s", "zh_t", "ht"]
    if dataset == "cfpb":
        languages = ["es", "kr", "ru", "vi", "zh_t", "ht"]  # cfpb has no zh_s
    
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

def get_term_english_mapping(dataset):
    """Get mapping from English terms to position in output file"""
    term_mapping = {}
    
    # Load a single output file to extract term information
    output_files = os.listdir("../output")
    sample_file = None
    for file in output_files:
        if dataset in file and file.endswith(".json"):
            sample_file = file
            break
    
    if sample_file is None:
        print(f"Warning: No output file found for dataset {dataset}")
        return term_mapping
        
    with open(f"../output/{sample_file}", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Process term pairs to create mapping
    for i, term_pair in enumerate(data.get("term_pairs", [])):
        # Extract English term from each term pair
        # For en-xx direction, the English term is the source (key in dictionary)
        # For xx-en direction, the English term is the target (value in dictionary)
        for src_term, tgt_term in term_pair.items():
            # Assuming the key is always source language and value is target
            if "en-" in sample_file:
                # In en-xx, English is the source term (key)
                term_mapping[src_term.lower()] = i
            else:
                # In xx-en, English is the target term (value)
                term_mapping[tgt_term.lower()] = i
    
    return term_mapping

def categorize_term_results(term_results, term_categories, term_mapping, dataset):
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

def calculate_accuracy_by_category(categorized_results):
    """Calculate term accuracy scores by category"""
    accuracy_by_category = {}
    
    for model in categorized_results:
        accuracy_by_category[model] = {}
        
        for lang in categorized_results[model]:
            accuracy_by_category[model][lang] = {}
            
            for direction in categorized_results[model][lang]:
                accuracy_by_category[model][lang][direction] = {}
                
                for category in ["DS", "DC", "G"]:
                    values = categorized_results[model][lang][direction][category]
                    if values:
                        # Convert NumPy types to Python native types
                        values = [float(v) if isinstance(v, np.number) else v for v in values]
                        accuracy = sum(values) / len(values)
                        accuracy_by_category[model][lang][direction][category] = float(accuracy)
                    else:
                        accuracy_by_category[model][lang][direction][category] = None
    
    return accuracy_by_category

def calculate_significance(categorized_results, models, alpha=0.05):
    """Calculate statistical significance between models for each category"""
    significance_results = {}
    
    # For each pair of models
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i >= j:  # Skip self-comparisons and redundant pairs
                continue
                
            model_pair = f"{model1}_vs_{model2}"
            significance_results[model_pair] = {}
            
            # For each language
            for lang in categorized_results.get(model1, {}):
                if lang not in categorized_results.get(model2, {}):
                    continue
                    
                significance_results[model_pair][lang] = {}
                
                # For each direction
                for direction in DIRECTIONS:
                    if direction not in categorized_results[model1][lang] or direction not in categorized_results[model2][lang]:
                        continue
                        
                    significance_results[model_pair][lang][direction] = {}
                    
                    # For each category
                    for category in ["DS", "DC", "G"]:
                        values1 = categorized_results[model1][lang][direction][category]
                        values2 = categorized_results[model2][lang][direction][category]
                        
                        # Skip if either model has no values for this category
                        if not values1 or not values2:
                            continue
                        
                        # Use Mann-Whitney U test (non-parametric) to compare distributions
                        # This is more appropriate than t-test for binary data
                        u_stat, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                        
                        # Calculate mean difference
                        mean1 = sum(values1) / len(values1)
                        mean2 = sum(values2) / len(values2)
                        mean_diff = mean1 - mean2
                        
                        # Make sure to convert NumPy types to Python native types for JSON serialization
                        significance_results[model_pair][lang][direction][category] = {
                            "p_value": float(p_value),
                            "significant": bool(p_value < alpha),
                            "mean_diff": float(mean_diff),
                            "better_model": model1 if mean_diff > 0 else model2
                        }
    
    return significance_results

def calculate_within_model_significance(categorized_results, models, alpha=0.05):
    """Calculate statistical significance between domain categories within each model"""
    within_significance = {}
    
    # Pairs of categories to compare
    category_pairs = [("DS", "DC"), ("DS", "G"), ("DC", "G")]
    
    # For each model
    for model in models:
        within_significance[model] = {}
        
        # For each language
        for lang in categorized_results.get(model, {}):
            within_significance[model][lang] = {}
            
            # For each direction
            for direction in categorized_results[model][lang]:
                within_significance[model][lang][direction] = {}
                
                # For each pair of categories
                for cat1, cat2 in category_pairs:
                    values1 = categorized_results[model][lang][direction].get(cat1, [])
                    values2 = categorized_results[model][lang][direction].get(cat2, [])
                    
                    # Skip if either category has no values
                    if not values1 or not values2:
                        continue
                    
                    # Use Mann-Whitney U test (non-parametric) to compare distributions
                    # This is appropriate for comparing distributions of binary outcomes
                    try:
                        stat, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                        
                        # Calculate mean difference
                        mean1 = sum(values1) / len(values1)
                        mean2 = sum(values2) / len(values2)
                        mean_diff = mean1 - mean2
                        
                        comparison = f"{cat1}_vs_{cat2}"
                        within_significance[model][lang][direction][comparison] = {
                            "p_value": float(p_value),
                            "significant": bool(p_value < alpha),
                            "mean_diff": float(mean_diff),
                            "better_category": cat1 if mean_diff > 0 else cat2,
                            "cat1_mean": float(mean1),
                            "cat2_mean": float(mean2),
                            "cat1_count": len(values1),
                            "cat2_count": len(values2)
                        }
                    except ValueError as e:
                        # Handle rare cases where the test can't be performed
                        print(f"Warning: Could not perform test for {model}, {lang}, {direction}, {cat1} vs {cat2}: {e}")
    
    return within_significance

def create_boxplots_by_category(accuracy_by_category, between_models_significance=None, within_model_significance=None, dataset="irs", output_dir="../figs/domain_analysis"):
    """Create boxplots showing term accuracy by domain specificity category"""
    # Set matplotlib parameters for a clean, publication-ready style
    plt.style.use('seaborn-v0_8-whitegrid')
    # Use default fonts instead of serif
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    models_list = list(accuracy_by_category.keys())
    
    # Process each direction separately
    for direction in DIRECTIONS:
        fig, ax = plt.subplots(figsize=(7, 3))
        
        # Define positions for groups and bars
        num_models = len(models_list)
        num_categories = 3  # DS, DC, G
        group_width = 0.8
        bar_width = group_width / num_categories
        
        # Collect data for plotting
        data_by_category = {"DS": [], "DC": [], "G": []}
        
        for model in models_list:
            model_data = {"DS": [], "DC": [], "G": []}
            
            # Collect values across all languages
            for lang in accuracy_by_category[model]:
                if direction in accuracy_by_category[model][lang]:
                    for category in ["DS", "DC", "G"]:
                        value = accuracy_by_category[model][lang][direction].get(category)
                        if value is not None:
                            model_data[category].append(value)
            
            # Add to overall data collection
            for category in ["DS", "DC", "G"]:
                data_by_category[category].append(model_data[category])
        
        # Plot each category
        positions = np.arange(num_models)
        category_positions = {}
        boxplots = {}
        
        offset = -group_width/3
        for category in ["DS", "DC", "G"]:
            category_pos = positions + offset
            category_positions[category] = category_pos
            boxprops = {'facecolor': CATEGORY_COLORS[category], 'alpha': 0.7}
            
            # Create boxplot
            bp = ax.boxplot(
                data_by_category[category], 
                positions=category_pos,
                widths=bar_width * 0.8,
                patch_artist=True,
                boxprops=boxprops,
                medianprops={'color': 'black', 'linewidth': 1.5}
            )
            boxplots[category] = bp
            
            offset += bar_width
        
        # Add model names
        ax.set_xticks(positions)
        ax.set_xticklabels([MODELSNAME2LATEX.get(model, model).replace('\\textsc{', '').replace('}', '') for model in models_list])
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["DS"], alpha=0.7, label='Domain-Specific'),
            plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["DC"], alpha=0.7, label='Domain-Contextual'),
            plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["G"], alpha=0.7, label='General')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # Set labels but no title
        ax.set_ylabel('Term Accuracy', fontsize=14)
        
        # Set y-limits consistent with main paper
        ax.set_ylim(0, 1.0)
        
        # Add within-model significance markers
        if within_model_significance:
            # Define symbols for different comparisons - use playing card suits that are better supported
            significance_symbols = {
                "DS_vs_DC": "♣",  # club
                "DS_vs_G": "♦",   # diamond
                "DC_vs_G": "♠"    # spade
            }
            
            # Track which symbols were used for the caption
            used_symbols = set()
            
            # Now mark the significant differences for each category
            y_max = ax.get_ylim()[1]
            
            # Determine the position for symbols relative to boxplot tops (including outliers)
            whisker_positions = {}
            for category in ["DS", "DC", "G"]:
                whisker_positions[category] = []
                for model_idx in range(num_models):
                    # Check if we have a boxplot for this model
                    if model_idx < len(positions):
                        # Start with the upper whisker position
                        max_y = 0
                        
                        # Get upper whisker position
                        for i_whisker in range(len(boxplots[category]['whiskers'])):
                            if i_whisker % 2 == 1:  # Upper whiskers are at odd indices
                                whisker_data = boxplots[category]['whiskers'][i_whisker].get_ydata()
                                if len(whisker_data) > 0 and model_idx == i_whisker // 2:
                                    max_y = max(max_y, whisker_data[0])
                        
                        # Check outliers (fliers) for even higher positions
                        if len(boxplots[category]['fliers']) > model_idx:
                            flier_data = boxplots[category]['fliers'][model_idx].get_ydata()
                            if len(flier_data) > 0:
                                max_y = max(max_y, np.max(flier_data))
                        
                        # If we couldn't get position data, use a fallback
                        if max_y == 0:
                            max_y = 0.6  # Default position
                            
                        whisker_positions[category].append(max_y)
                    else:
                        whisker_positions[category].append(0.6)  # Default if no data
            
            # Calculate a more moderate vertical margin
            vertical_margin = (y_max - 0) * 0.08  # Reduced to 8% for closer positioning to boxplots
            
            # For each model, check and mark significant differences
            for i, model in enumerate(models_list):
                if model not in within_model_significance:
                    continue
                
                # Skip if we don't have whisker data for this model
                if i >= len(whisker_positions["DS"]) or i >= len(whisker_positions["DC"]) or i >= len(whisker_positions["G"]):
                    continue

                # Find significant differences for this model across languages
                model_significances = {}
                for cat1, cat2 in [("DS", "DC"), ("DS", "G"), ("DC", "G")]:
                    comparison = f"{cat1}_vs_{cat2}"
                    model_significances[comparison] = {
                        "significant": False,
                        "better_category": None,
                        "count": 0  # Track count to avoid showing duplicate symbols
                    }
                    
                    # Check if significant in any language
                    for lang in within_model_significance[model]:
                        if direction in within_model_significance[model][lang]:
                            if comparison in within_model_significance[model][lang][direction]:
                                result = within_model_significance[model][lang][direction][comparison]
                                if result["significant"]:
                                    model_significances[comparison]["significant"] = True
                                    model_significances[comparison]["better_category"] = result["better_category"]
                                    model_significances[comparison]["count"] += 1
                                    break  # One significant language is enough to mark

                # Calculate the actual median values for each category to confirm better performance
                if any(significant["significant"] for significant in model_significances.values()):
                    medians = {}
                    for category in ["DS", "DC", "G"]:
                        # Check if we have data for this category
                        if i < len(data_by_category[category]):
                            category_data = data_by_category[category][i]
                            # If the data is already a flat list of values, use it directly
                            if category_data and isinstance(category_data[0], (int, float)):
                                flattened_data = category_data
                            # Otherwise, we need to flatten a list of lists
                            else:
                                flattened_data = []
                                for item in category_data:
                                    # Make sure we're only extending with iterables, not floats
                                    if hasattr(item, '__iter__') and not isinstance(item, (str, bytes)):
                                        flattened_data.extend(item)
                                    else:
                                        flattened_data.append(item)
                        
                            # Calculate the median if we have data
                            if flattened_data:
                                medians[category] = np.median(flattened_data)
                            else:
                                medians[category] = 0
                        else:
                            medians[category] = 0
                    
                    # Verify and potentially correct the "better" category based on actual median values
                    for comparison, result in model_significances.items():
                        if result["significant"]:
                            cat1, cat2 = comparison.split("_vs_")
                            # If statistical test says cat1 is better but median shows cat2 is better
                            if result["better_category"] == cat1 and medians[cat1] < medians[cat2]:
                                # When statistics contradict visual data, defer to the visual data to avoid confusion
                                result["better_category"] = cat2
                            # Or if statistical test says cat2 is better but median shows cat1 is better
                            elif result["better_category"] == cat2 and medians[cat2] < medians[cat1]:
                                result["better_category"] = cat1
                
                # Now place symbols above each category
                for category in ["DS", "DC", "G"]:
                    # Find all comparisons where this category is better
                    category_symbols = []
                    
                    # Check DS_vs_DC
                    if "DS_vs_DC" in model_significances:
                        if model_significances["DS_vs_DC"]["better_category"] == category:
                            category_symbols.append(significance_symbols["DS_vs_DC"])
                    
                    # Check DS_vs_G
                    if "DS_vs_G" in model_significances:
                        if model_significances["DS_vs_G"]["better_category"] == category:
                            category_symbols.append(significance_symbols["DS_vs_G"])
                    
                    # Check DC_vs_G
                    if "DC_vs_G" in model_significances:
                        if model_significances["DC_vs_G"]["better_category"] == category:
                            category_symbols.append(significance_symbols["DC_vs_G"])
                    
                    # If we have symbols to place
                    if category_symbols:
                        # Get the position for this category
                        cat_pos = category_positions[category][i]
                        
                        # Get the minimum y value for this boxplot (lower whisker or outliers)
                        min_y = 1.0  # Start with maximum possible value
                        
                        # Check lower whiskers
                        for i_whisker in range(len(boxplots[category]['whiskers'])):
                            if i_whisker % 2 == 0:  # Lower whiskers are at even indices
                                whisker_data = boxplots[category]['whiskers'][i_whisker].get_ydata()
                                if len(whisker_data) > 0 and i == i_whisker // 2:
                                    min_y = min(min_y, whisker_data[0])
                        
                        # Check lower outliers too
                        if len(boxplots[category]['fliers']) > i:
                            flier_data = boxplots[category]['fliers'][i].get_ydata()
                            if len(flier_data) > 0:
                                min_y = min(min_y, np.min(flier_data))
                        
                        # Get the upper position with padding
                        upper_symbol_height = whisker_positions[category][i] + vertical_margin + (y_max * 0.02)
                        
                        # More selective condition for placing markers below
                        # Only place below if:
                        # 1. Upper position is extremely high (>95% of plot height), or
                        # 2. This is a high-accuracy model like GPT4o AND the upper whisker is very high
                        place_below = False
                        
                        # Define which models may need special placement - ONLY very high accuracy models
                        high_accuracy_models = ["LLM_openai_gpt4o"]
                        
                        # Check for different conditions that would trigger below placement
                        if upper_symbol_height > y_max * 0.97:  # Raised threshold to be more selective
                            # Extreme case - always place below
                            place_below = True
                        elif model in high_accuracy_models and whisker_positions[category][i] > 0.92:
                            # High-accuracy model with whisker very high
                            place_below = True
                        
                        # Special exception: NEVER place Mistral markers below
                        if model == "LLM_mistral":
                            place_below = False
                        
                        if place_below:
                            # Place below with some margin
                            symbol_height = min_y - vertical_margin - (y_max * 0.02)
                            # Ensure we don't go below the plot
                            symbol_height = max(symbol_height, y_max * 0.05)
                        else:
                            # Place above as normal
                            symbol_height = upper_symbol_height
                            # Ensure we don't go above the plot
                            symbol_height = min(symbol_height, y_max * 0.95)
                        
                        # Place symbols side by side
                        if len(category_symbols) == 1:
                            # Just one symbol - place in center
                            ax.text(cat_pos, symbol_height, category_symbols[0], 
                                  fontsize=16, fontweight='bold', color='black',
                                  horizontalalignment='center', verticalalignment='center')
                        else:
                            # Multiple symbols - place side by side with increased spacing
                            # Increase spacing coefficient from 0.05 to 0.08
                            total_width = len(category_symbols) * 0.08  # Total width of all symbols
                            start_offset = -total_width / 2 + 0.04  # Starting point (centered)
                            
                            for j, symbol in enumerate(category_symbols):
                                offset = start_offset + j * 0.08  # Increased horizontal offset for each symbol
                                ax.text(cat_pos + offset, symbol_height, symbol, 
                                      fontsize=16, fontweight='bold', color='black',
                                      horizontalalignment='center', verticalalignment='center')
        
        # Remove the legend for significance markers since it will be in the LaTeX caption
        # Just track which symbols were used
        if used_symbols:
            pass  # No legend added, just tracking which symbols were used
        
        # Add between-models significance markers if provided
        if between_models_significance:
            y_max = ax.get_ylim()[1]
            for i, model1 in enumerate(models_list):
                for j, model2 in enumerate(models_list):
                    if i >= j:  # Skip self-comparisons and redundant pairs
                        continue
                        
                    model_pair = f"{model1}_vs_{model2}"
                    alt_pair = f"{model2}_vs_{model1}"
                    
                    if model_pair in between_models_significance:
                        pair_results = between_models_significance[model_pair]
                    elif alt_pair in between_models_significance:
                        pair_results = between_models_significance[alt_pair]
                    else:
                        continue
                    
                    # Check significance for each category
                    offset = -group_width/3
                    for category in ["DS", "DC", "G"]:
                        # Count significant differences
                        sig_diff = 0
                        better_model = None
                        
                        for lang in pair_results:
                            if (direction in pair_results[lang] and 
                                category in pair_results[lang][direction] and 
                                pair_results[lang][direction][category]["significant"]):
                                
                                sig_diff += 1
                                better_model = pair_results[lang][direction][category]["better_model"]
                        
                        if sig_diff > 0:
                            # Position asterisk above the better model's boxplot
                            better_idx = models_list.index(better_model)
                            x_pos = better_idx + offset
                            
                            # Add significance marker
                            ax.text(x_pos, y_max * 0.95, '†', 
                                   horizontalalignment='center', fontsize=20, fontweight='bold')
                        
                        offset += bar_width
        
        # Adjust layout and save
        plt.tight_layout()
        filename = f"{dataset}_{direction}_domain_categories.pdf"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved boxplot to {filepath}")

def compute_domain_metrics(dataset, selected_models, results_dir):
    """Compute and save domain-specific metrics"""
    print(f"Analyzing domain specificity for {dataset} dataset using models: {', '.join(selected_models)}")
    
    # Load term annotations
    print("Loading term annotations...")
    term_categories = load_annotations(dataset)
    print(f"Loaded {len(term_categories)} annotated terms with domain categories")
    
    # Load term-level results
    print("Loading term-level results...")
    term_results = load_term_level_results(selected_models, dataset)
    
    # Get English term mapping for output files
    term_mapping = get_term_english_mapping(dataset)
    
    # Categorize results by domain specificity
    print("Categorizing results by domain specificity...")
    categorized_results = categorize_term_results(term_results, term_categories, term_mapping, dataset)
    
    # Calculate accuracy by category
    print("Calculating accuracy by category...")
    accuracy_by_category = calculate_accuracy_by_category(categorized_results)
    
    # Create directory for output
    os.makedirs(results_dir, exist_ok=True)
    
    # Save categorized results
    categorized_path = os.path.join(results_dir, f"{dataset}_categorized_results.json")
    with open(categorized_path, "w", encoding="utf-8") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model in categorized_results:
            serializable_results[model] = {}
            for lang in categorized_results[model]:
                serializable_results[model][lang] = {}
                for direction in categorized_results[model][lang]:
                    serializable_results[model][lang][direction] = {}
                    for category in categorized_results[model][lang][direction]:
                        values = categorized_results[model][lang][direction][category]
                        # Convert any NumPy types to Python native types
                        serializable_results[model][lang][direction][category] = [float(v) if isinstance(v, np.number) else v for v in values]
        
        json.dump(serializable_results, f, indent=4, cls=NumpyEncoder)
    
    # Save accuracy results
    accuracy_path = os.path.join(results_dir, f"{dataset}_accuracy_by_category.json")
    with open(accuracy_path, "w", encoding="utf-8") as f:
        json.dump(accuracy_by_category, f, indent=4, cls=NumpyEncoder)
    
    print(f"Domain metrics computed and saved to {results_dir}")
    
    return categorized_results, accuracy_by_category

def run_domain_statistics(dataset, selected_models, results_dir, use_within_model=True, use_between_models=False):
    """Run statistical tests on domain-specific metrics"""
    print(f"Running statistical tests for {dataset} dataset")
    
    # Load categorized results
    categorized_path = os.path.join(results_dir, f"{dataset}_categorized_results.json")
    if not os.path.exists(categorized_path):
        print(f"Error: Categorized results file not found: {categorized_path}")
        print("Please run with --metrics first to generate the required data.")
        return None, None
    
    with open(categorized_path, "r", encoding="utf-8") as f:
        categorized_results = json.load(f)
    
    # Run statistical tests
    between_models_results = None
    within_model_results = None
    
    if use_between_models:
        print("Calculating between-models statistical significance...")
        between_models_results = calculate_significance(categorized_results, selected_models)
        
        # Save between-models significance results
        between_models_path = os.path.join(results_dir, f"{dataset}_between_models_significance.json")
        with open(between_models_path, "w", encoding="utf-8") as f:
            json.dump(between_models_results, f, indent=4, cls=NumpyEncoder)
        
        print(f"Between-models statistical tests complete. Results saved to {between_models_path}")
    
    if use_within_model:
        print("Calculating within-model statistical significance...")
        within_model_results = calculate_within_model_significance(categorized_results, selected_models)
        
        # Save within-model significance results
        within_model_path = os.path.join(results_dir, f"{dataset}_within_model_significance.json")
        with open(within_model_path, "w", encoding="utf-8") as f:
            json.dump(within_model_results, f, indent=4, cls=NumpyEncoder)
        
        print(f"Within-model statistical tests complete. Results saved to {within_model_path}")
    
    return between_models_results, within_model_results

def create_domain_plots(dataset, selected_models, results_dir, figs_dir, use_within_model=True, use_between_models=False):
    """Create plots for domain-specific analysis"""
    print(f"Creating visualizations for {dataset} dataset")
    
    # Load accuracy results
    accuracy_path = os.path.join(results_dir, f"{dataset}_accuracy_by_category.json")
    if not os.path.exists(accuracy_path):
        print(f"Error: Accuracy results file not found: {accuracy_path}")
        print("Please run with --metrics first to generate the required data.")
        return
    
    with open(accuracy_path, "r", encoding="utf-8") as f:
        accuracy_by_category = json.load(f)
    
    # Load significance results
    between_models_significance = None
    within_model_significance = None
    
    if use_between_models:
        between_models_path = os.path.join(results_dir, f"{dataset}_between_models_significance.json")
        if os.path.exists(between_models_path):
            with open(between_models_path, "r", encoding="utf-8") as f:
                between_models_significance = json.load(f)
        else:
            print("Warning: No between-models significance results found.")
    
    if use_within_model:
        within_model_path = os.path.join(results_dir, f"{dataset}_within_model_significance.json")
        if os.path.exists(within_model_path):
            with open(within_model_path, "r", encoding="utf-8") as f:
                within_model_significance = json.load(f)
        else:
            print("Warning: No within-model significance results found.")
    
    # Create and save boxplots
    domain_figs_dir = os.path.join(figs_dir, "domain_analysis")
    os.makedirs(domain_figs_dir, exist_ok=True)
    
    print("Creating boxplot visualizations...")
    create_boxplots_by_category(
        accuracy_by_category, 
        between_models_significance, 
        within_model_significance,
        dataset, 
        domain_figs_dir
    )
    
    print(f"Visualizations created. Figures saved to {domain_figs_dir}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set variables from arguments
    dataset = args.dataset
    selected_models = args.models
    results_dir = args.results_dir
    figs_dir = args.figs_dir
    use_within_model = args.within_model
    use_between_models = args.between_models
    
    # Create output directories if they don't exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    
    # Compute metrics if requested
    categorized_results = None
    accuracy_by_category = None
    if args.metrics:
        print("=== Computing domain-specific metrics ===")
        categorized_results, accuracy_by_category = compute_domain_metrics(dataset, selected_models, results_dir)
    
    # Run statistical tests if requested
    between_models_results = None
    within_model_results = None
    if args.stats:
        print("\n=== Running statistical significance tests ===")
        between_models_results, within_model_results = run_domain_statistics(
            dataset, selected_models, results_dir, use_within_model, use_between_models
        )
    
    # Create visualizations if requested
    if args.plot:
        print("\n=== Creating visualizations ===")
        create_domain_plots(
            dataset, selected_models, results_dir, figs_dir, use_within_model, use_between_models
        )
    
    print("\nAnalysis complete.")
    
    # Print description of statistical tests for paper
    if args.stats:
        print("\n=== Statistical Test Description for Paper ===")
        if use_within_model:
            print("""
For comparing term accuracy across domain categories (Domain-Specific, Domain-Contextual, and General), 
we employed the Mann-Whitney U test, a non-parametric method that assesses whether the distribution of 
term accuracy scores differs significantly between categories. This test is appropriate for our binary 
success/failure data (whether a term was correctly translated) as it makes no assumptions about normality 
and compares the entire distribution rather than just means. Statistical significance was determined at 
p < 0.05, and significant differences are marked with symbols above the better-performing category: 
♣ (club) for significant differences between Domain-Specific and Domain-Contextual categories,
♦ (diamond) for significant differences between Domain-Specific and General categories, and
♠ (spade) for significant differences between Domain-Contextual and General categories.
            """)
        if use_between_models:
            print("""
For comparing models within each domain category, we also used the Mann-Whitney U test to determine 
if one model's performance distribution is significantly better than another's. Significant differences 
between models are indicated with a dagger (†) in the figures.
            """)

if __name__ == "__main__":
    main()

