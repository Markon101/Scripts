import os
from Crypto.Cipher import AES

def gen_aes(key_len=32, block_len=16):
    key = os.urandom(key_len)
    iv = os.urandom(block_len)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return iv + cipher.encrypt(os.urandom(1000 * 1024 * 1024))

def gen_aes_file(filename="test.txt", key_len=32, block_len=16, file_size_mb=1024):
    key_and_ciphertext = gen_aes(key_len, block_len)
    with open(filename, "wb") as f:
        f.write(key_and_ciphertext)

def main():
    filename = "test.txt"
    key_len = 32
    block_len = 16
    gen_aes_file(filename=filename, key_len=key_len, block_len=block_len)

if __name__ == "__main__":
    main()

