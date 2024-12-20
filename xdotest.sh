#!/bin/bash

# List all xterm windows
WINDOW_IDS=$(xdotool search --name "xterm")

if [ -z "$WINDOW_IDS" ]; then
    echo "No xterm windows found."
    exit 0
fi

echo "List of all xterm windows:"

# Loop through each window ID and get details
for ID in $WINDOW_IDS
do
    WINDOW_TITLE=$(xdotool getwindowname "$ID")
    echo "Window ID: $ID, Title: $WINDOW_TITLE"
done

