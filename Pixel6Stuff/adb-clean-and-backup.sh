#!/bin/bash

# Define the backup directory
BACKUP_DIR="/mnt/external1/pixel6_backup_$(date +'%Y%m%d')"
BACKUP_FILE="$BACKUP_DIR/backup.ab"

# Check if ADB is installed
if ! command -v adb &> /dev/null; then
    echo "ADB not installed. Please install ADB and try again."
    exit 1
fi

# Create the backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

echo "Backing up your Pixel 6 running Android 15 BETA..."

# Ensure the device is connected
adb wait-for-device

# Clear app caches without deleting data
echo "Clearing all app caches..."
adb shell pm list packages | cut -d ':' -f 2 | while read -r package; do
    adb shell pm clear "$package" --user 0
    echo "Cleared cache for $package"
done

# Compile and optimize all apps
echo "Optimizing all apps..."
adb shell cmd package compile -m speed-profile -f -a
echo "All apps optimized."

# Backup system and user data with encryption
echo "Backing up system and user data with encryption..."
adb backup -apk -obb -shared -all -system -f "$BACKUP_FILE" -encrypt

# Check if the backup file was created successfully
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Backup failed. Exiting..."
    exit 1
fi

# Compress the backup for easier storage
echo "Compressing the backup..."
tar -czf "$BACKUP_DIR.tar.gz" -C "$BACKUP_DIR" "$(basename "$BACKUP_FILE")"

# Cleanup: Remove the uncompressed backup directory
rm -rf "$BACKUP_DIR"

echo "Backup and encryption complete! The backup is saved as $BACKUP_DIR.tar.gz."

exit 0

