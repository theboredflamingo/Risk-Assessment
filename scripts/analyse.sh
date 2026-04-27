#!/bin/bash

echo "Recent runs:"
tail -n 5 run_log.json

echo ""
echo "Errors in pipeline:"
grep -i error logs/pipeline.log