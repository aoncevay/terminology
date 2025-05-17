#!/bin/bash

# This script runs the sacrebleu paired bootstrap resampling test for chrF++ scores
# It uses MADLAD as the baseline model and compares all other models against it

# Variables
models=("MADLAD" "NLLB")  # MADLAD is the baseline
datasets=("irs" "cfpb")
languages=("es" "kr" "ru" "vi" "zh_s" "zh_t" "ht")
output_dir="../output"
results_dir="../results"
temp_dir="/tmp/chrf_bootstrap"

# Create temp directory if it doesn't exist
mkdir -p $temp_dir
mkdir -p $results_dir

# Loop through datasets
for dataset in "${datasets[@]}"; do
    echo "Processing $dataset dataset..."
    
    # Combined output file for this dataset
    combined_output_file="$results_dir/chrf_bootstrap_${dataset}_ALL.txt"
    > "$combined_output_file"  # Clear or create file
    
    # Loop through language pairs
    for lang in "${languages[@]}"; do
        for direction in "$lang-en" "en-$lang"; do
            echo "  Processing language pair: $direction"
            
            # Check which models have data for this language pair
            available_models=()
            baseline_exists=false
            
            # First, find available models and extract translation data
            for model in "${models[@]}"; do
                output_file="$output_dir/$model.$dataset.$direction.json"
                
                if [ -f "$output_file" ]; then
                    available_models+=("$model")
                    
                    # Extract reference and hypothesis translations using jq (if installed)
                    if command -v jq &> /dev/null; then
                        jq -r '.ref | join("\n")' "$output_file" > "$temp_dir/ref.$direction.txt"
                        jq -r '.hyp | join("\n")' "$output_file" > "$temp_dir/$model.$direction.txt"
                    else
                        # Fallback using Python (assumes Python is installed)
                        python -c "
import json
with open('$output_file', 'r') as f:
    data = json.load(f)
with open('$temp_dir/ref.$direction.txt', 'w') as f:
    f.write('\n'.join(data['ref']))
with open('$temp_dir/$model.$direction.txt', 'w') as f:
    f.write('\n'.join(data['hyp']))
"
                    fi
                    
                    # Check if baseline model exists
                    if [ "$model" == "MADLAD" ]; then
                        baseline_exists=true
                    fi
                fi
            done
            
            # If we have the baseline and at least one other model, run the test
            if [ "$baseline_exists" = true ] && [ ${#available_models[@]} -gt 1 ]; then
                echo "    Running bootstrap test with models: ${available_models[*]}"
                
                # Prepare input files for sacrebleu
                input_files=()
                for model in "${available_models[@]}"; do
                    input_files+=("$temp_dir/$model.$direction.txt")
                done
                
                # Run sacrebleu with paired bootstrap
                echo "    Executing sacrebleu command..."
                
                # Construct command for better readability
                cmd=(
                    "sacrebleu"
                    "--input" "${input_files[@]}"
                    "--reference" "$temp_dir/ref.$direction.txt"
                    "--metrics" "chrf"
                    "--paired-bs"
                )
                
                # Execute and save to output file
                direction_output_file="$results_dir/chrf_bootstrap_${dataset}_${direction}.txt"
                
                if "${cmd[@]}" > "$direction_output_file" 2>&1; then
                    echo "    Success! Results saved to $direction_output_file"
                    
                    # Add to combined output
                    {
                        echo -e "\n\n================================================================================"
                        echo "RESULTS FOR $direction (${available_models[*]})"
                        echo -e "================================================================================"
                        cat "$direction_output_file"
                    } >> "$combined_output_file"
                else
                    echo "    Error running sacrebleu command. Check $direction_output_file for details."
                    
                    # Add error to combined output
                    {
                        echo -e "\n\nERROR FOR $direction:"
                        echo "Command: ${cmd[*]}"
                        echo "See $direction_output_file for details."
                    } >> "$combined_output_file"
                fi
            else
                echo "    Skipping $direction: Missing baseline model or only one model available"
                
                # Add note to combined output
                {
                    echo -e "\n\nSKIPPED $direction:"
                    echo "Available models: ${available_models[*]}"
                    echo "Baseline model (MADLAD) exists: $baseline_exists"
                } >> "$combined_output_file"
            fi
        done
    done
    
    echo "Completed bootstrap analysis for $dataset dataset"
    echo "Combined results saved to $combined_output_file"
done

echo "All bootstrap analyses completed."
echo "Results are in $results_dir/chrf_bootstrap_*"

# Clean up temp files (optional)
# rm -rf "$temp_dir" 