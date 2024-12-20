#!/bin/bash

# Backup current iptables rules before making changes
iptables-save > ~/iptables-backup-$(date +%F).rules

# Set default policies
# We set the default policy for INPUT to DROP to ensure that by default, no unsolicited incoming traffic is allowed.
iptables -P INPUT DROP
# We set the default policy for FORWARD to DROP because this machine is not a router; it should not forward packets by default.
iptables -P FORWARD DROP
# We set the default policy for OUTPUT to ACCEPT to allow all outgoing traffic from this machine.
iptables -P OUTPUT ACCEPT

# Allow incoming SSH connections on port 3579
# This rule specifically allows incoming TCP packets on port 3579, which we are using for SSH in this case.
iptables -A INPUT -p tcp --dport 3579 -j ACCEPT

# Allow established and related incoming connections
# This rule allows incoming traffic that is a response to traffic sent from this machine, enabling two-way communication for established sessions.
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Save the new rules to ensure they persist across reboots
# On Arch Linux, iptables rules are saved to /etc/iptables/iptables.rules by default.
iptables-save | sudo tee /etc/iptables/iptables.rules

# Restart iptables to apply changes
# This ensures that the iptables service picks up the changes we've made.
sudo systemctl restart iptables

# List the new iptables rules
# It's always good practice to list the rules after modification to ensure they have been applied correctly.
iptables -L -v -n

