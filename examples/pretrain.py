import torch
import argparse
from hypertext.models.llama_proto import LlamaProto
from hypertext.training.trainer import HyperTrainer
from hypertext.data.dataset import create_dataloader
from hypertext.utils.distributed import DistEnv

def main():
    parser = argparse.ArgumentParser(description="HyperText Pretraining")
    parser.add_argument("--data_path", type=str, default="data.bin", help="Path to binary dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--name", type=str, default="hyper_llama_nano")
    args = parser.parse_args()
    
    # 1. Environment Init
    if DistEnv.is_main_process():
        print(f"Starting Training: {args.name}")
        print(f"World Size: {DistEnv.get_world_size()}")
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. Data Loading
    # Auto-creates dummy data if missing
    train_loader = create_dataloader(args.data_path, block_size=128, batch_size=args.batch_size)
    
    # 3. Model Init
    model = LlamaProto(
        vocab_size=args.vocab_size, 
        d_model=args.dim, 
        n_layers=args.n_layers, 
        n_heads=args.n_heads
    )
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 4. Trainer Init
    trainer = HyperTrainer(
        model=model,
        train_loader=train_loader,
        device=device,
        mixed_precision=(device == 'cuda'),
        grad_accum_steps=4, # Simulate large batch training
        checkpoint_dir='checkpoints'
    )
    
    # 5. Execute
    trainer.fit(epochs=args.epochs)
    
    if DistEnv.is_main_process():
        print("Training Job Completed Successfully.")

if __name__ == "__main__":
    main()
