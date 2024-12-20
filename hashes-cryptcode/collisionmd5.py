import numpy as np

# The MD5 hash
hash_value = "367f214089b68ac3936e6a17da39c5b3"

# Convert each character in the hash to its ASCII value
ascii_values = [ord(char) for char in hash_value]

# Create a 4x8 matrix from the ASCII values
matrix_representation = np.array(ascii_values).reshape(4,8)
matrix_representation = matrix_representation.flatten()
matrix_representation = matrix_representation.ravel()

print("Matrix representation of the MD5 hash:")
print(matrix_representation)

