import random

# Fallback word list if dictionary file is not found
fallback_words = ["apple", "banana", "cherry", "date", "elderberry", "fig", 
                  "grape", "honeydew", "iceberg", "jackfruit", "kiwi", 
                  "lemon", "mango", "nectarine", "orange", "papaya", 
                  "quince", "raspberry", "strawberry", "tangerine", "ugli", 
                  "vanilla", "watermelon", "xigua", "yellowfruit", "zucchini"]

# Load the dictionary file
def load_dictionary(file_path):
    try:
        with open(file_path, 'r') as file:
            words = file.read().splitlines()
        return words
    except FileNotFoundError:
        print("Dictionary file not found. Using fallback word list.")
        return fallback_words

# Generate a random seed from the dictionary
def generate_seed(words, num_words=500):
    if len(words) < num_words:
        print("Not enough words in the dictionary to generate the seed.")
        return ""
    
    selected_words = random.sample(words, num_words)
    seed = ' '.join(selected_words)
    return seed

# Path to your dictionary file (this is just a placeholder)
dictionary_file_path = '/home/anon/Downloads/words.txt'

# Load the dictionary
word_list = load_dictionary(dictionary_file_path)

# Number of words to include in the seed
NUM_WORDS = 500

# Generate the seed
if word_list:
    seed = generate_seed(word_list, NUM_WORDS)
    print("Generated seed:", seed)

