#!/usr/bin/env python3
import os
import json
import argparse
import sacrebleu
from sacrebleu.metrics import CHRF

def process_output_file(input_file, output_file):
    """
    Process an MT output file, compute per-sentence chrF++ and term accuracy,
    and transform the data structure from dict of lists to list of dicts.
    
    Args:
        input_file: Path to input MT output file (e.g., output/LLM.aya.irs.en-ru.json)
        output_file: Path to output file
    """
    print(f"Processing {input_file}")
    
    # Load the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize chrF++ metric calculator
    chrf_metric = CHRF(word_order=2)  # chrF++
    
    # Transform the data structure
    processed_entries = []
    
    # Check if the required fields exist in the data
    required_fields = ["doc_id", "src", "ref", "hyp", "term_pairs"]
    for field in required_fields:
        if field not in data:
            print(f"Error: Required field '{field}' not found in the input file")
            return
    
    # Get the length of the arrays (assuming all arrays have the same length)
    num_entries = len(data["doc_id"])
    
    # Process each entry
    for i in range(num_entries):
        # Extract fields for this entry
        doc_id = data["doc_id"][i]
        src = data["src"][i]
        ref = data["ref"][i]
        hyp = data["hyp"][i]
        term_pairs = data["term_pairs"][i]
        
        # Compute chrF++ for this sentence
        chrf_score = chrf_metric.sentence_score(hypothesis=hyp, references=[ref]).score
        
        # Compute term accuracy for this sentence
        term_count = 0
        term_correct = 0
        for term, translation in term_pairs.items():
            term_count += 1
            if translation.lower() in hyp.lower():
                term_correct += 1
        
        term_accuracy = term_correct / term_count if term_count > 0 else 0.0
        
        # Create entry dictionary
        entry = {
            "doc_id": doc_id,
            "src": src,
            "ref": ref,
            "hyp": hyp,
            "term_pairs": term_pairs,
            "chrF++": chrf_score,
            "term_accuracy": term_accuracy
        }
        
        processed_entries.append(entry)
    
    # Save the processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_entries, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {num_entries} entries, saved to {output_file}")
    print(f"Sample entry: {processed_entries[0] if processed_entries else 'No entries'}")


def main():
    parser = argparse.ArgumentParser(description="Process MT output files for qualitative analysis")
    
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input MT output file (e.g., output/LLM.aya.irs.en-ru.json)")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output file")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Process the file
    process_output_file(args.input, args.output)


if __name__ == "__main__":
    main() 