import tiktoken

class Tokenizer:
    def __init__(self):
        # Load OpenAI's GPT-4 BPE tokenizer
        self.enc = tiktoken.get_encoding("cl100k_base")
        
        # The vocabulary size of this tokenizer is exactly 100,277
        self.vocab_size = self.enc.n_vocab
        
    def encode(self, text):
        """Takes a string, outputs a list of integers"""
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})
        
    def decode(self, tokens):
        """Takes a list of integers, outputs a string"""
        return self.enc.decode(tokens)

# Create a global instance so other files can import `encoder.vocab_size` easily
encoder = Tokenizer()
