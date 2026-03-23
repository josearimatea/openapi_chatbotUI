# Function que save and load chunks using pickle

import pickle

# Function to save chunks to a file
def save_chunks(chunks, filename):
    with open(filename, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to {filename}")

# Function to load chunks from a file
def load_chunks(filename):
    with open(filename, 'rb') as f:
        chunks = pickle.load(f)
    print(f"Chunks loaded from {filename}")
    return chunks