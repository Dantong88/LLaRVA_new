#!/bin/bash

source activate llarva
echo "-----------------------------------------"
echo "In Conda envrironment:" $CONDA_DEFAULT_ENV
echo "-----------------------------------------"

echo "-----------------------------------------"
wandb-osh -- --sync-all