#!/bin/bash

if [ -f app.pid ]; then
    kill -9 $(cat app.pid)
    rm app.pid
    echo "App stopped"
else
    echo "No running app found"
fi