#!/bin/bash

# Define the arrays for models and normalization types
models=("bert" "distilbert" "roberta")
#norms=("sign_norm" "sign_magl2" "sign_magl1" "safe_norm" "square_norm")
norms=( "safe_norm")

# Start the nested loop
for model_name in "${models[@]}"; do
    for norm_name in "${norms[@]}"; do
        
        echo "------------------------------------------------"
        echo "Running: Model=$model_name | GNorm=$norm_name"
        echo "------------------------------------------------"
        
        # Execute the python command
        python run_eval_pg_sentiment_gnorm.py \
            --model "$model_name" \
            --dataset sst2 \
            --gnorm "$norm_name"
            
        # Optional: check if the previous command failed
        if [ $? -ne 0 ]; then
            echo "Error occurred with $model_name using $norm_name. Continuing..."
        fi

    done
done

echo "Evaluation suite finished successfully!"