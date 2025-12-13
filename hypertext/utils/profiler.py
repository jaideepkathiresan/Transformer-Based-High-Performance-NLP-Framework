import torch
from contextlib import contextmanager

@contextmanager
def profile_scope(name, enabled=True, export_path="trace.json"):
    """
    Context manager for PyTorch Profiler.
    """
    if enabled and torch.cuda.is_available():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            yield prof
    else:
        # No-op listener for CPU-only or disabled
        yield None
