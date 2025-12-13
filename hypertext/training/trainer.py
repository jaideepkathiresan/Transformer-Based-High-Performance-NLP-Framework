import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time

class HyperTrainer:
    def __init__(
        self,
        model,
        train_loader,
        eval_loader=None,
        optimizer=None,
        scheduler=None,
        device='cpu',
        grad_accum_steps=1,
        mixed_precision=False,
        checkpoint_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.mixed_precision = mixed_precision
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.optimizer = optimizer or optim.AdamW(model.parameters(), lr=1e-4)
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision and device == 'cuda' else None
        
    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        
        self.optimizer.zero_grad()
        
        for i, batch in enumerate(self.train_loader):
            # unpack batch (assuming dict or tuple)
            if isinstance(batch, dict):
                x = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
            else:
                x, labels = batch
                x = x.to(self.device)
                labels = labels.to(self.device)
                
            # Mixed Precision Context
            with torch.cuda.amp.autocast(enabled=self.mixed_precision and self.device == 'cuda'):
                outputs = self.model(x)
                
                # Handle different output types (tuple vs tensor)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                    
                # Reshape for Loss: (B*Seq, Vocab)
                B, S, V = logits.shape
                loss = nn.functional.cross_entropy(logits.view(-1, V), labels.view(-1))
                loss = loss / self.grad_accum_steps
                
            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            total_loss += loss.item() * self.grad_accum_steps
            
            # Step
            if (i + 1) % self.grad_accum_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                    
                self.optimizer.zero_grad()
                
            if i % 10 == 0:
                print(f"Epoch {epoch_idx} | Step {i} | Loss: {loss.item() * self.grad_accum_steps:.4f}")
                
        avg_loss = total_loss / len(self.train_loader)
        duration = time.time() - start_time
        print(f"Epoch {epoch_idx} Complete. Avg Loss: {avg_loss:.4f}. Time: {duration:.2f}s")
        return avg_loss

    def save_checkpoint(self, name):
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved: {path}")

    def fit(self, epochs):
        print(f"Starting Training for {epochs} epochs...")
        for epoch in range(epochs):
            self.train_epoch(epoch)
            self.save_checkpoint(f"epoch_{epoch}")
