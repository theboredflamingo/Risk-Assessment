#!/bin/bash

cd /mnt/c/Users/viole/stocks || exit

source stockenv/bin/activate

mkdir -p logs

python main.py >> logs/pipeline.log 2>&1