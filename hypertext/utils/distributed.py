import os
import torch

class DistEnv:
    """
    Abstractions for Distributed Training Environment.
    Handles extraction of Rank, World Size, and Local Rank from env vars.
    """
    @staticmethod
    def get_rank():
        return int(os.getenv('RANK', '0'))
        
    @staticmethod
    def get_world_size():
        return int(os.getenv('WORLD_SIZE', '1'))
        
    @staticmethod
    def get_local_rank():
        return int(os.getenv('LOCAL_RANK', '0'))
    
    @staticmethod
    def is_main_process():
        return DistEnv.get_rank() == 0

def barrier():
    """
    Synchronization barrier for distributed processes.
    """
    if DistEnv.get_world_size() > 1:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

def all_reduce(tensor, op='sum'):
    """
    Perform all-reduce operation across distributed ranks.
    """
    if DistEnv.get_world_size() > 1 and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor
