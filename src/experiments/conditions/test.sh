#!/bin/bash

# absolute path to this script
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
EXPERIMENT_DIR="src/experiments/conditions"

# script's parent directory
SCRIPT_DIR="$SCRIPT_PATH"

# get the project main directory
for i in $(seq 1 4);
do
    SCRIPT_DIR="$(dirname "$SCRIPT_DIR")"
done

# change directory to project main directory
cd $SCRIPT_DIR

# create logging directory
LOGGING_DIR="${EXPERIMENT_DIR}/logging"
if [ ! -d $LOGGING_DIR ]; then
    mkdir $LOGGING_DIR
fi

# create score directory
SCORE_DIR="${EXPERIMENT_DIR}/output"
if [ ! -d $SCORE_DIR ]; then
    mkdir $SCORE_DIR
fi

# run script
python -m src.scripts.test_conditions \
    --config $EXPERIMENT_DIR/test_config.yml \
    --data_path data/experimental_condition/LLPSDB2_conditions_LAF1_RGG.csv \
    --score_path $EXPERIMENT_DIR/output/LLPSDB2_conditions_LAF1_RGG_score.csv \
    > $EXPERIMENT_DIR/logging/LLPSDB2_conditions_LAF1_RGG.log
