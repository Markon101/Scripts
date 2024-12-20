#!/bin/bash

# Define a word list with 16 entries (0-15)
word_list=("apple" "banana" "cherry" "date" "elderberry" "fig" "grape" "honeydew" \
           "iceberg" "jackfruit" "kiwi" "lemon" "mango" "nectarine" "orange" "papaya")

# Number of words to generate in the seed
NUM_WORDS=8

# Generate a random SHA-512 hash
random_hash=$(dd if=/dev/random bs=1M count=1 2>/dev/null | sha512sum | awk '{print $1}')

# Convert the first few characters of the hash into words
seed=""
for ((i=0; i<$NUM_WORDS; i++))
do
    # Extract the i-th hexadecimal digit from the hash
    hex_digit="${random_hash:$i:1}"
    
    # Convert the hex digit to a decimal value
    decimal_value=$((16#$hex_digit))
    
    # Map the decimal value to a word in the word list
    word=${word_list[$decimal_value]}
    
    # Append the word to the seed
    seed+="$word "
done

# Output the generated seed value
echo "Generated seed: $seed"

