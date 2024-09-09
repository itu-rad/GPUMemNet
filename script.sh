#!/bin/bash

# Define ranges for the random parameters
depth_min=2
depth_max=6

width_min=64
width_max=256

# Define batch sizes as powers of two
batch_sizes=(16 32 64 128)

datasets=("mnist" "cifar10" "cifar100" "imagenet")

# Create a log file to track the progress
log_file="training_log.txt"
echo "Training Log - $(date)" > $log_file

# Loop for 2000 iterations to generate random parameters and train
for i in {1..2000}
do
  # Randomly generate depth, width
  depth=$((RANDOM % (depth_max - depth_min + 1) + depth_min))
  width=$((RANDOM % (width_max - width_min + 1) + width_min))
  
  # Randomly select a batch size from the predefined list of powers of two
  batch_size=${batch_sizes[$RANDOM % ${#batch_sizes[@]}]}
  
  # Randomly select a dataset from the available choices
  dataset=${datasets[$RANDOM % ${#datasets[@]}]}

  echo "Training iteration $i with the following parameters:" | tee -a $log_file
  echo "Depth: $depth" | tee -a $log_file
  echo "Width: $width" | tee -a $log_file
  echo "Batch Size: $batch_size" | tee -a $log_file
  echo "Dataset: $dataset" | tee -a $log_file

  # Call the Python script with the generated parameters and log errors if any
  if ! python3 mlp.py --depth $depth --width $width --batch_size $batch_size --dataset $dataset >> $log_file 2>&1; then
    echo "Iteration $i failed with Depth: $depth, Width: $width, Batch Size: $batch_size, Dataset: $dataset" | tee -a $log_file
  fi

  echo "Iteration $i complete." | tee -a $log_file
done