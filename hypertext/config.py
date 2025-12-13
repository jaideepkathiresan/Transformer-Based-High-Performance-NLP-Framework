from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    vocab_size: int = 32000
    max_seq_len: int = 2048
    dropout: float = 0.1
    use_moe: bool = False
    num_experts: int = 8
    
@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 100000
    grad_accum_steps: int = 1
    mixed_precision: bool = True
    output_dir: str = "checkpoints"
