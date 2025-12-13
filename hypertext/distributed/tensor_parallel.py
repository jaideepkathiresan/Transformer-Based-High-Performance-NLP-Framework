import torch
import torch.nn as nn
import torch.nn.functional as F
from hypertext.utils.distributed import DistEnv, all_reduce

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with weight split along the column axis.
    Used for the First layer in an FFN or QKV projection.
    """
    def __init__(self, in_features, out_features, bias=True, gather_output=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        
        world_size = DistEnv.get_world_size()
        self.output_size_per_partition = out_features // world_size
        
        # Note: In a real TP system, we'd initialize with a seed to sync, then split.
        self.weight = nn.Parameter(torch.randn(self.output_size_per_partition, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.output_size_per_partition))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        # x: (Batch, ..., In)
        # Weight: (Out/N, In)
        # Result: (Batch, ..., Out/N)
        output = F.linear(x, self.weight, self.bias)
        
        if self.gather_output and DistEnv.get_world_size() > 1:
            # All-Gather logic would go here to reconstruct full output
            pass
            
        return output

class RowParallelLinear(nn.Module):
    """
    Linear layer with weight split along the row axis.
    Used for the Second layer in an FFN or Output projection.
    Requires an All-Reduce on the output.
    """
    def __init__(self, in_features, out_features, bias=True, input_is_parallel=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        
        world_size = DistEnv.get_world_size()
        self.input_size_per_partition = in_features // world_size
        
        # Weight: (Out, In/N)
        self.weight = nn.Parameter(torch.randn(out_features, self.input_size_per_partition))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        # x: (Batch, ..., In/N) if input_is_parallel
        # Weight: (Out, In/N)
        # Result: (Batch, ..., Out) partial sum
        
        output_parallel = F.linear(x, self.weight)
        
        # All-Reduce (Sum) across ranks to get final result
        output = all_reduce(output_parallel, op='sum')
        
        if self.bias is not None:
            output = output + self.bias
            
        return output
