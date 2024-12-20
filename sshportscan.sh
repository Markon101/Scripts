#!/bin/bash

# Set common username and password (leave empty for no authentication)
username="admin"
password="passw0rd"

# Check if a remote host is provided, otherwise use localhost as the target
if [ -n "$1" ]; then
  target="$1"
else
  target="192.168.2.0/24"
fi

# Scan for open ports on $target using Nmap
nmap_output=$(sudo nmap -sS --open -p 22,80 $target | grep -oP '(?<=PORT\s.*?/\s).*?')
echo "Scanning $target..."
if [ ! -z "$nmap_output" ]; then
  echo "Found open ports on $target:"
  echo "$nmap_output"
else
  echo "$(date): No open ports found on $target."
fi

# Attempt to log in to discovered SSH servers using common username/password or no authentication
if [ ! -z "$nmap_output" ]; then
  for server in $(echo "$nmap_output" | tr ',' '\n'); do
    if [[ "$server" =~ ^(?:127\.0\.0\.1|)?([a-zA-Z0-9\.]+)\.local$ ]]; then
      ip="$1"
    else
      ip="$(echo "$server" | awk '{print $1}')"
    fi

    if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 22 "${username}"@$ip; then
      echo "SUCCESS: Logged in to $server using common credentials."
    else
      echo "FAILED: Unable to log in to $server due to incorrect username/password or no authentication available."
    fi
  done
fi

echo "$(date): Script completed. No open ports found on $target, and SSH servers were not accessed with common credentials."

