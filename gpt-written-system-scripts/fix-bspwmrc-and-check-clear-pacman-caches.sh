#!/bin/bash

# Function to clean up package caches
cleanup_package_caches() {
    echo "Cleaning up package caches..."
    sudo pacman -Scc --noconfirm
    sudo pacman -Rns $(pacman -Qdtq) --noconfirm
    echo "Package caches cleaned."
}

# Function to check and attempt to repair bspwmrc configuration
check_bspwmrc() {
    BSPWMRC_PATH="$HOME/.config/bspwm/bspwmrc"
    BACKUP_PATH="$HOME/.config/bspwm/bspwmrc.backup"
    REPAIR_PATH="$HOME/.config/bspwm/bspwmrc.repair"

    echo "Checking bspwmrc configuration..."

    # Check if bspwmrc file exists
    if [ ! -f "$BSPWMRC_PATH" ]; then
        echo "bspwmrc file not found at $BSPWMRC_PATH"
        exit 1
    fi

    # Make a backup copy of bspwmrc
    cp "$BSPWMRC_PATH" "$BACKUP_PATH"
    cp "$BSPWMRC_PATH" "$REPAIR_PATH"

    # Function to perform detailed checks
    perform_detailed_checks() {
        local config_file="$1"
        local is_valid=true

        # Check for shebang
        if ! grep -q "^#!/bin/bash" "$config_file"; then
            echo "Adding missing shebang."
            echo "#!/bin/bash" | cat - "$config_file" > temp && mv temp "$config_file"
        fi

        # Ensure file ends with a newline
        sed -i -e '$a\' "$config_file"

        # Check for basic bspwm commands
        local commands=("bspc config" "bspc monitor" "bspc rule" "bspc subscribe")
        for cmd in "${commands[@]}"; do
            if ! grep -q "$cmd" "$config_file"; then
                echo "Warning: '$cmd' command not found in bspwmrc."
                is_valid=false
            fi
        done

        # Add default monitor configuration if missing
        if ! grep -q "bspc monitor" "$config_file"; then
            echo "Adding default monitor configuration."
            echo "bspc monitor -d I II III IV V VI VII VIII IX X" >> "$config_file"
        fi

        # Add default border width configuration if missing
        if ! grep -q "bspc config border_width" "$config_file"; then
            echo "Adding default border width configuration."
            echo "bspc config border_width 2" >> "$config_file"
        fi

        # Add default window rules if missing
        if ! grep -q "bspc rule" "$config_file"; then
            echo "Adding default window rules."
            echo "bspc rule -a Gimp desktop=\^8 follow=on state=floating" >> "$config_file"
            echo "bspc rule -a Chromium desktop=\^2" >> "$config_file"
        fi

        # Add example rule for terminals
        if ! grep -q "bspc rule -a Alacritty" "$config_file"; then
            echo "Adding rule for Alacritty terminal."
            echo "bspc rule -a Alacritty state=tiled" >> "$config_file"
        fi

        # Validate the final file
        if bash -n "$config_file"; then
            echo "bspwmrc syntax is correct."
        else
            echo "bspwmrc syntax is still incorrect after repairs."
            is_valid=false
        fi

        echo "$is_valid"
    }

    # Perform detailed checks on the original bspwmrc
    if bash -n "$BSPWMRC_PATH"; then
        echo "bspwmrc syntax is correct."
        if [ "$(perform_detailed_checks "$BSPWMRC_PATH")" == "true" ]; then
            echo "bspwmrc configuration is valid."
        else
            echo "bspwmrc configuration has warnings. Check the log for details."
        fi
    else
        echo "bspwmrc configuration has syntax errors. Attempting to create a repaired copy..."

        # Perform detailed checks on the repaired bspwmrc
        if [ "$(perform_detailed_checks "$REPAIR_PATH")" == "true" ]; then
            echo "Repaired bspwmrc configuration is valid."
        else
            echo "Repaired bspwmrc configuration still has errors."
        fi

        echo "Repaired bspwmrc saved as $REPAIR_PATH"
    fi
}

# Main script execution
cleanup_package_caches
check_bspwmrc

echo "Script execution completed."

