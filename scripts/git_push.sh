#!/bin/bash

echo "[INFO] Pushing updates..."

git config --global user.email "stuthi.shrisha@github.com"
git config --global user.name "theboredflamingo"

git add dailym.pkl run_log.json
git commit -m "auto update $(date)"
git push origin main