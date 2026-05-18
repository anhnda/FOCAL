#!/bin/bash

# Define the list of unique scripts to run
scripts=(
    #"run_eval_attcat_sentiment.py"
    "run_eval_ig_sentiment.py"
    "run_eval_pg_sentiment.py"
    #"run_slalom_eval_sentiment.py"
    #"run_eval_reagent_sentiment.py"
)

# Define the list of models
models=("bert" "roberta" "distilbert")

# Define the list of possible methods for evaluation
methods=("mask")

# 1. Iterate through each script
for script in "${scripts[@]}"; do
    # 2. Iterate through each model
    for model in "${models[@]}"; do
        # 3. Iterate through each evaluation baseline method
        for eval_baseline in "${methods[@]}"; do
            
            echo "=========================================================="
            echo "SCRIPT:   $script"
            echo "MODEL:    $model"
            echo "EVAL:     $eval_baseline"
            echo "=========================================================="
            
            # Execute the python script
            # Removed the --baseline flag as requested
            python "$script" \
                --model "$model" \
                --eval-baseline "$eval_baseline" \
                --dataset sst2
            
            # Check if the command succeeded
            if [ $? -ne 0 ]; then
                echo "ERROR: $script failed for $model (Eval Baseline: $eval_baseline)"
            fi
            
            echo -e "\n"
        done
    done
done

echo "All tests completed across all scripts and models."