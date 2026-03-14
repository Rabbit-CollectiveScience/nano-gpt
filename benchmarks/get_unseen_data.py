import os
import urllib.request
import torch

# Ensure we can import our tokenizer from the root folder
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from shared.step1_tokenizer import encoder

def download_and_tokenize():
    print("Downloading pristine benchmark dataset...")
    # We use Alice in Wonderland from Project Gutenberg as our "Unseen" test data
    url = "https://www.gutenberg.org/files/11/11-0.txt"
    file_path = os.path.join(current_dir, 'alice.txt')
    
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded to {file_path}")
    else:
        print(f"File {file_path} already exists.")

    print("Tokenizing the dataset...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # We skip the heavy Project Gutenberg header/footer text mathematically
    start_idx = text.find("Down the Rabbit-Hole")
    end_idx = text.find("*** END OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***")
    clean_text = text[start_idx:end_idx] if start_idx != -1 else text

    # Tokenize and save as a PyTorch tensor
    data = torch.tensor(encoder.encode(clean_text), dtype=torch.long)
    tensor_path = os.path.join(current_dir, 'alice.pt')
    torch.save(data, tensor_path)
    
    print(f"Successfully saved {len(data)} test tokens to {tensor_path}!")

if __name__ == '__main__':
    download_and_tokenize()
