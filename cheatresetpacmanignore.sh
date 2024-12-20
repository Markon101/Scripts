#!/bin/bash

# Backup the original pacman.conf
sudo cp /etc/pacman.conf /etc/pacman.conf.bak

# Function to remove IgnorePkg and IgnoreGroup directives
remove_ignore_directives() {
  sudo sed -i '/^IgnorePkg/d' /etc/pacman.conf
  sudo sed -i '/^IgnoreGroup/d' /etc/pacman.conf
}

# Function to verify the changes
verify_changes() {
  if ! grep -q '^IgnorePkg' /etc/pacman.conf && ! grep -q '^IgnoreGroup' /etc/pacman.conf; then
    echo "Successfully removed IgnorePkg and IgnoreGroup directives."
  else
    echo "Failed to remove IgnorePkg and/or IgnoreGroup directives."
  fi
}

# Execute the functions
remove_ignore_directives
verify_changes

