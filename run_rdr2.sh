#!/bin/bash

# Stop the display manager
#echo "Stopping SDDM..."
#sudo systemctl stop sddm

# Set CPU governor to performance mode
echo "Setting CPU governor to performance..."
sudo cpupower frequency-set -g performance

# Configure NVIDIA GPU performance mode (if using NVIDIA)
if command -v nvidia-settings &> /dev/null; then
    echo "Setting NVIDIA GPU to Performance Mode..."
    nvidia-settings -a '[gpu:0]/GpuPowerMizerMode=1'
fi

# Set display resolution to 4K
#echo "Setting display resolution to 4K on HDMI..."
#xrandr --output HDMI --mode 3840x2160

# Launch Steam in offline mode with GameMode for performance
echo "Launching Red Dead Redemption 2 via Steam in offline mode with GameMode..."
gamemoderun steam -offline steam://rungameid/1174180 &

# Wait for user to close the game
#echo "Waiting for the game to close..."
#wait

# Re-enable the display manager after game closes
#echo "Restarting SDDM..."
#sudo systemctl start sddm

# Reset CPU governor to default
#echo "Resetting CPU governor to default mode..."
#sudo cpupower frequency-set -g ondemand

#echo "Script completed. SDDM and CPU governor restored."

