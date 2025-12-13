import argparse
import os
import numpy as np
from tqdm import tqdm
# In a real scenario, we'd use 'tiktoken' or 'transformers'. 
# For zero-dependency demo, we use a simple char-level or whitespace tokenizer.

def simple_tokenizer(text):
    # Mock Tokenizer: maps unique chars to ints
    # In production: replace with BPE
    return [ord(c) % 32000 for c in text]

def process_file(input_path, output_path, chunk_size=1024*1024):
    print(f"Processing {input_path} -> {output_path}")
    
    # Pre-calculate size or grow dynamically? 
    # For mmap, we usually pre-allocate or write sequentially then append.
    # We'll use a simple append strategy with file write.
    
    total_tokens = 0
    with open(output_path, 'wb') as f_out:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                    
                tokens = simple_tokenizer(chunk)
                arr = np.array(tokens, dtype=np.uint16)
                f_out.write(arr.tobytes())
                total_tokens += len(tokens)
                
    print(f"Detailed: Written {total_tokens} tokens to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Data Preparation Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument("--output", type=str, default="data.bin", help="Output binary file")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        # Create dummy input if missing
        print("Input file not found. Creating dummy 'corpus.txt'...")
        with open(args.input, 'w', encoding='utf-8') as f:
            f.write("HyperText Infinite is the future of NLP. " * 10000)
            
    process_file(args.input, args.output)

if __name__ == "__main__":
    main()
