import torch
import torch.nn as nn
import torch.optim as optim
from hypertext.model import HyperBert

def train_demo():
    print("Initializing HyperBert Training Demo...")
    
    # Hyperparameters
    vocab_size = 1000
    d_model = 256
    seq_len = 32
    batch_size = 16
    
    # Model
    model = HyperBert(vocab_size, d_model=d_model, n_layers=2, max_len=seq_len)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy Data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size * seq_len,)) # flattened labels
    
    model.train()
    print("Starting training loop...")
    for epoch in range(3):
        optimizer.zero_grad()
        
        output = model(input_ids) # (Batch, Seq, Vocab)
        output_flat = output.view(-1, vocab_size)
        
        loss = criterion(output_flat, labels)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
        
    print("Training demo completed successfully!")

if __name__ == "__main__":
    train_demo()
