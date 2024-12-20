#!/bin/bash

# Function to clear yay cache
clear_yay_cache() {
    echo "Clearing yay cache..."
    yay -Sc --noconfirm
}

# Function to clear pacman cache
clear_pacman_cache() {
    echo "Clearing pacman cache..."
    sudo pacman -Sc --noconfirm
}

# Function to clear Mamba/Conda cache
clear_conda_cache() {
    echo "Clearing Mamba/Conda cache..."
    mamba clean --all --yes
}

# Function to clear pip cache
clear_pip_cache() {
    echo "Clearing pip cache..."
    pip cache purge
}

# Execute functions
clear_yay_cache
clear_pacman_cache
clear_conda_cache
clear_pip_cache

echo "All caches cleared successfully!"

