#!/bin/bash

# Get the list of all external USB drives
external_drives=$(lsblk -o name,tran | awk '$2=="usb" {print $1}' | xargs -I {} sudo blkid /dev/{}1 | awk -F'"' '{print $10}')

# Initialize the mount count
mount_count=1

# Loop through the list of external USB drives and mount them
for drive in $external_drives; do
  # Get the UUID of the drive
  drive_uuid=$drive

  # Create the mount point
  mount_point="/mnt/external$mount_count"
  mkdir -p $mount_point

  # Check if the drive is already mounted
  if ! grep -qs "$mount_point" /proc/mounts; then
    # Mount the drive and tell systemd to automount when available
    mount -t auto -o rw,relatime PARTUUID=$drive_uuid $mount_point
    echo "Drive mounted at $mount_point"
    systemctl start media-$USER-$drive_uuid.mount.service

    # Increment the mount count
    mount_count=$((mount_count + 1))
  else
    echo "Drive already mounted at $mount_point"
  fi
done

# Output information to the terminal when -v is passed as a parameter.
if [[ "$1" == "-v" ]]; then
  echo "All external USB drives have been mounted."
fi


