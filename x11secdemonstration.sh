#!/bin/bash

# Demonstration of security issues with X11

# Check if xterm is installed
if ! command -v xterm &> /dev/null
then
    echo "xterm could not be found, please install it and try again."
    exit 1
fi

# Set the DISPLAY variable to the local display (usually :0)
export DISPLAY=:0

# Open a new xterm window
xterm -hold -e "echo 'This is a demonstration of X11 security issues'; bash" &

# Wait for the xterm window to open
sleep 2

# Find the window ID of the xterm window
XTERM_WINDOW_ID=$(xdotool search --name "alacritty")

if [ -z "$XTERM_WINDOW_ID" ]; then
    echo "Could not find the xterm window."
    exit 1
fi

# Simulate typing in the xterm window
xdotool type --window "$XTERM_WINDOW_ID" "echo 'X11 allows remote control of GUI applications'; bash"
xdotool key --window "$XTERM_WINDOW_ID" Return

echo "A new xterm window has been opened and a message has been typed into it."

# Note: The xdotool commands above simulate typing into the xterm window.
# This demonstrates how X11 can be used to control GUI applications.

# Clean up (optional): Close the xterm window after a delay
sleep 10
xdotool windowclose "$XTERM_WINDOW_ID"

echo "The xterm window has been closed."

