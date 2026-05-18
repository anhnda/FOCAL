#!/bin/bash

# Define the list of possible values
methods=("pad" "zero" "mean")

# Iterate through each combination
for baseline in "${methods[@]}"; do
    for eval_baseline in "${methods[@]}"; do
        
        echo "=========================================================="
        echo "RUNNING: --baseline $baseline --eval-baseline $eval_baseline"
        echo "=========================================================="
        
        # Execute the python script
        python run_eval_pg_gpt2.py --baseline "$baseline" --eval-baseline "$eval_baseline"
        
        # Check if the command succeeded
        if [ $? -ne 0 ]; then
            echo "Error occurred with baseline: $baseline and eval: $eval_baseline"
        fi
        
        echo -e "\n"
    done
done

echo "All tests completed."