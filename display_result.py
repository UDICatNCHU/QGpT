#!/usr/bin/env python3
"""Extract recall metrics from BGE M3 experiment JSON files."""

import json
from pathlib import Path


def extract_recall_metrics(experiment_dir: str = "bge_m3_flag/experiment") -> None:
    """Extract and display recall metrics from all JSON files in directory.
    
    Args:
        experiment_dir: Directory containing JSON experiment files
    """
    experiment_path = Path(experiment_dir)
    
    if not experiment_path.exists():
        print(f"Directory {experiment_dir} not found")
        return
    
    json_files = list(experiment_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {experiment_dir}")
        return
    
    for json_file in sorted(json_files):
        print(f"========== {json_file.name} =======")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract recall metrics with default fallback
            metrics = [
                ("avg_recall_at_1", data.get("avg_recall_at_1", "N/A")),
                ("avg_recall_at_3", data.get("avg_recall_at_3", "N/A")),
                ("avg_recall_at_5", data.get("avg_recall_at_5", "N/A")),
                ("avg_recall_at_10", data.get("avg_recall_at_10", "N/A"))
            ]
            
            for metric_name, value in metrics:
                print(f'  "{metric_name}": {value},')
                
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Error reading file: {e}")
        
        print()  # Empty line between files


if __name__ == "__main__":
    extract_recall_metrics("experiment/")