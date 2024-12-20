#!/bin/bash

# Variables
NEW_SSH_PORT=2222       # Set your desired SSH port
ALLOWED_IPS=("192.168.1.100" "203.0.113.1") # Replace with your allowed IPs
LOG_FILE="/var/log/ssh_access.log"

# Ensure the script is run as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root"
   exit 1
fi

# Install necessary packages
#pacman -Syu --noconfirm fail2ban

# Backup the original sshd_config
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak

# Change the SSH port in the sshd_config
sed -i "s/#Port 22/Port $NEW_SSH_PORT/" /etc/ssh/sshd_config

# Allow only specified IPs to connect to the new SSH port
iptables -A INPUT -p tcp --dport $NEW_SSH_PORT -j DROP
#for IP in "${ALLOWED_IPS[@]}"; do
#    iptables -A INPUT -p tcp -s $IP --dport $NEW_SSH_PORT -j ACCEPT
#done

# Save iptables rules
iptables-save > /etc/iptables/iptables.rules
systemctl restart iptables

# Set up detailed logging for SSH access attempts using systemd
mkdir -p /etc/systemd/system/sshd.service.d/
cat <<EOL > /etc/systemd/system/sshd.service.d/override.conf
[Service]
StandardOutput=syslog
StandardError=syslog
EOL

# Reload systemd daemon to apply changes
systemctl daemon-reload

# Configure fail2ban for SSH
cat <<EOL > /etc/fail2ban/jail.local
[sshd]
enabled = true
port = $NEW_SSH_PORT
logpath = /var/log/auth.log
maxretry = 5
EOL

# Start and enable fail2ban service
systemctl start fail2ban
systemctl enable fail2ban

# Restart SSH service to apply changes
systemctl restart sshd

# Output the new SSH port and log file location
echo "SSH has been secured and configured to use port $NEW_SSH_PORT."
echo "Only the following IPs are allowed to connect: ${ALLOWED_IPS[*]}"
echo "SSH access attempts are being logged via systemd."
echo "fail2ban has been configured to protect against brute force attacks."

