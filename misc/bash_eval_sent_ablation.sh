#!/bin/bash

# Define the arrays
models=("bert" "distilbert" "roberta")
methods=("l1" "l2" "scalar")

# Outer loop for models
for model_name in "${models[@]}"; do
    # Inner loop for methods
    for method_name in "${methods[@]}"; do
        
        echo "================================================"
        echo "MODEL: $model_name | METHOD: $method_name"
        echo "================================================"
        
        python run_eval_pg_sentiment_ablation.py \
            --dataset sst2 \
            --model "$model_name" \
            --method "$method_name"
            
    done
done

echo "All combinations of models and methods are complete!"