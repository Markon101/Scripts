#!/bin/bash

# Run the Python script to generate the AES encrypted file
python3 aes_generator.py

# Define file names
aes_file="test.txt"
zip_file="conatainers.zip"
encrypted_zip_file="encrypted.zip.enc"

# Zip the AES encrypted file
zip -r "$zip_file" "$aes_file"

# Generate a random password
password=$(openssl rand -base64 32)

# Encrypt the zip file with the random password
openssl enc -aes-256-cbc -in "$zip_file" -out "$encrypted_zip_file" -pass pass:"$password"

echo "Zip file encrypted with password: $password"

