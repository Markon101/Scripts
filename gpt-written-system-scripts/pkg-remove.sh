#!/bin/bash

# ============================================
# Package Cleaner Script for Multiple Distros
# ============================================

# 1. Detect the distribution type
get_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    else
        echo "unknown"
    fi
}

distro=$(get_distro)
echo "Detected distro: $distro"

# 2. Check if deborphan is installed on Ubuntu/Debian
check_deborphan() {
    if ! command -v deborphan &> /dev/null; then
        echo "deborphan is not installed. Installing deborphan..."
        sudo apt update && sudo apt install -y deborphan
        if ! command -v deborphan &> /dev/null; then
            echo "Failed to install deborphan. Exiting."
            exit 1
        fi
    fi
}

# 3. List installed packages
list_installed_packages() {
    case "$distro" in
        fedora|rhel|centos)
            # Skip the header line
            dnf list installed | awk 'NR>1 {print $1}'
            ;;
        ubuntu|debian)
            dpkg-query -W -f='${binary:Package}\n'
            ;;
        arch)
            pacman -Qq
            ;;
        *)
            echo "Unsupported distribution"
            exit 1
            ;;
    esac
}

# 4. Check if a package is depended on by others
check_if_depended_on() {
    package=$1
    case "$distro" in
        fedora|rhel|centos)
            dependencies=$(dnf repoquery --whatrequires "$package" | grep -v "^$")
            ;;
        ubuntu|debian)
            dependencies=$(apt-cache rdepends "$package" | tail -n +2) # Skip first line (package itself)
            ;;
        arch)
            dependencies=$(pacman -Qi "$package" | grep "Required By" | sed 's/Required By\s*:\s*//')
            # If 'Required By' is 'none' or empty, set dependencies to empty
            if [[ "$dependencies" == "none" || "$dependencies" == "None" || -z "$dependencies" ]]; then
                dependencies=""
            fi
            ;;
        *)
            echo "Unsupported distribution"
            exit 1
            ;;
    esac

    if [[ -n "$dependencies" ]]; then
        echo "$package is required by other packages:"
        echo "$dependencies"
        return 1
    else
        echo "$package is not required by any other package."
        return 0
    fi
}

# 5. Find orphaned packages
find_orphaned_packages() {
    case "$distro" in
        fedora|rhel|centos)
            dnf repoquery --unneeded
            ;;
        ubuntu|debian)
            check_deborphan
            deborphan
            ;;
        arch)
            pacman -Qdtq
            ;;
        *)
            echo "Unsupported distribution"
            exit 1
            ;;
    esac
}

# 6. Find large packages (only package names, sorted descending by size)
find_large_packages() {
    case "$distro" in
        fedora|rhel|centos)
            dnf repoquery --installed --qf "%{size}\t%{name}" | sort -nr | awk '{print $2}'
            ;;
        ubuntu|debian)
            dpkg-query -W --showformat='${Installed-Size}\t${Package}\n' | sort -nr | awk '{print $2}'
            ;;
        arch)
            # Parse Installed Size and convert to KiB for accurate sorting
            pacman -Qi | awk '
                /^Name/ {name=$3}
                /^Installed Size/ {
                    size=$4
                    unit=$5
                    if(unit == "KiB") {
                        size_kib = size
                    } else if(unit == "MiB") {
                        size_kib = size * 1024
                    } else if(unit == "GiB") {
                        size_kib = size * 1024 * 1024
                    } else {
                        size_kib = 0
                    }
                    if(size_kib > 0) {
                        print size_kib "\t" name
                    }
                }
                /^$/ {name=""; size_kib=""}
            ' | sort -nr | awk '{print $2}'
            ;;
        *)
            echo "Unsupported distribution"
            exit 1
            ;;
    esac
}

# 7. Find unused packages
find_unused_packages() {
    case "$distro" in
        fedora|rhel|centos)
            # This heuristic may not be precise
            dnf history userinstalled | awk 'NR>1 {print $1}'
            ;;
        ubuntu|debian)
            apt-mark showauto
            ;;
        arch)
            pacman -Qetq
            ;;
        *)
            echo "Unsupported distribution"
            exit 1
            ;;
    esac
}

# 8. Remove package safely with confirmation and dependency check
remove_package_safely() {
    package=$1
    check_if_depended_on "$package"
    if [[ $? -eq 0 ]]; then
        echo "Do you want to remove '$package'? (yes/no)"
        read -r confirm
        if [[ "$confirm" == "yes" ]]; then
            case "$distro" in
                fedora|rhel|centos)
                    sudo dnf remove -y "$package"
                    ;;
                ubuntu|debian)
                    sudo apt remove -y "$package"
                    ;;
                arch)
                    sudo pacman -Rns "$package"
                    ;;
                *)
                    echo "Unsupported distribution"
                    exit 1
                    ;;
            esac
            if [[ $? -eq 0 ]]; then
                echo "Successfully removed '$package'."
            else
                echo "Failed to remove '$package'."
            fi
        else
            echo "Skipped removal of '$package'."
        fi
    else
        echo "Cannot remove '$package' as it is required by other packages."
    fi
}

# 9. Collect results and prompt for each package
collect_and_prompt() {
    echo "Finding orphaned packages..."
    orphans=$(find_orphaned_packages)
    echo "Finding large packages..."
    large_packages=$(find_large_packages)
    echo "Finding unused packages..."
    unused_packages=$(find_unused_packages)

    # Combine all packages, deduplicate
    combined_packages=$(echo -e "$orphans\n$large_packages\n$unused_packages" | sort | uniq)

    # Check if there are any packages to process
    if [[ -z "$combined_packages" ]]; then
        echo "No candidate packages found for removal."
        exit 0
    fi

    echo -e "\nCandidate packages for removal:"
    echo "$combined_packages"

    # Iterate over each package
    for package in $combined_packages; do
        # Skip empty entries
        if [[ -z "$package" ]]; then
            continue
        fi
        # Remove possible version numbers or invalid names by verifying installation
        if case "$distro" in
            fedora|rhel|centos) dnf list installed "$package" &> /dev/null ;;
            ubuntu|debian) dpkg-query -W -f='${binary:Package}\n' "$package" &> /dev/null ;;
            arch) pacman -Qi "$package" &> /dev/null ;;
            *) false ;;
        esac; then
            remove_package_safely "$package"
        else
            echo "Package '$package' not found or not installed. Skipping."
        fi
    done
}

# 10. Run the process
collect_and_prompt

