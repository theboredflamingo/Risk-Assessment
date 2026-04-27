#!/bin/bash

echo "[INFO] Starting Streamlit app..."

source venv/bin/activate

nohup streamlit run trynew.py > logs/app.log 2>&1 &
echo $! > app.pid