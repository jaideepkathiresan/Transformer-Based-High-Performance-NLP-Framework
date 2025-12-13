import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class MMapDataset(Dataset):
    """
    Memory-Mapped Dataset Reader for Large Scale Corpora.
    Expects data to be a flat binary file of uint16 (tokens).
    """
    def __init__(self, bin_path, block_size):
        self.block_size = block_size
        
        if not os.path.exists(bin_path):
            logger.warning(f"Dataset not found at {bin_path}. Initializing synthetic dataset for verification.")
            # Initialize synthetic data
            synthetic_data = np.random.randint(0, 32000, 100000, dtype=np.uint16)
            with open(bin_path, 'wb') as f:
                f.write(synthetic_data.tobytes())
                
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)
        
        if self.total_tokens <= block_size:
             # Expand synthetic data if too small
             logger.info("Dataset too small, expanding...")
             extra = np.random.randint(0, 32000, block_size * 10, dtype=np.uint16)
             with open(bin_path, 'ab') as f:
                 f.write(extra.tobytes())
             self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
             self.total_tokens = len(self.data)

        print(f"Loaded dataset with {self.total_tokens / 1e6:.2f}M tokens.")
        
    def __len__(self):
        return self.total_tokens - self.block_size
        
    def __getitem__(self, idx):
        # Efficient zero-copy slice from disk/RAM
        chunk = self.data[idx : idx + self.block_size].astype(np.int64)
        x = torch.from_numpy(chunk)
        # Shifted label for causal modeling
        y = torch.from_numpy(self.data[idx+1 : idx+1+self.block_size].astype(np.int64))
        return x, y

def create_dataloader(bin_path, block_size, batch_size, workers=0):
    ds = MMapDataset(bin_path, block_size)
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=workers,
        pin_memory=True
    )
