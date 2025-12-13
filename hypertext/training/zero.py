import torch
from hypertext.utils.distributed import DistEnv

class ZeroOptimizer:
    """
    ZeRO Stage 1 Simulator: Shards optimizer state.
    """
    def __init__(self, params, optimizer_cls, mp_world_size=1, **kwargs):
        self.params = list(params)
        self.world_size = DistEnv.get_world_size() # simulated
        self.rank = DistEnv.get_rank()
        
        # In real ZeRO, we split params into partitions.
        # Here we verify the logic:
        total_params = len(self.params)
        partition_size = total_params // self.world_size
        
        start_idx = self.rank * partition_size
        end_idx = start_idx + partition_size if self.rank != self.world_size - 1 else total_params
        
        self.my_params = self.params[start_idx:end_idx]
        
        # Only initialize optimizer for MY partition
        self.optimizer = optimizer_cls(self.my_params, **kwargs)
        
    def step(self):
        # 1. All-Reduce Gradients (conceptually)
        # 2. Optimizer Step on partition
        self.optimizer.step()
        
        # 3. All-Gather updated params (conceptually)
        pass
        
    def zero_grad(self):
        self.optimizer.zero_grad()
