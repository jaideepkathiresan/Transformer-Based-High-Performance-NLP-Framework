import unittest
import torch
import sys
import os

# Ensure import path is correct
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hypertext.models.llama_proto import LlamaProto
from hypertext.models.moe import MoELayer
from hypertext.distributed.tensor_parallel import ColumnParallelLinear

class TestModels(unittest.TestCase):
    def test_llama_forward(self):
        print("\nTesting LLaMA Model Forward Pass...")
        vocab_size = 100
        d_model = 64
        n_layers = 2
        n_heads = 4
        
        model = LlamaProto(vocab_size, d_model, n_layers, n_heads)
        input_ids = torch.randint(0, vocab_size, (2, 10)) # Batch=2, Seq=10
        
        output = model(input_ids)
        
        self.assertEqual(output.shape, (2, 10, vocab_size))
        self.assertFalse(torch.isnan(output).any(), "Output contains NaNs")
        print("LLaMA Forward: OK")

    def test_moe_forward(self):
        print("\nTesting MoE Layer Forward Pass...")
        d_model = 32
        d_ff = 128
        num_experts = 4
        
        moe = MoELayer(d_model, d_ff, num_experts)
        x = torch.randn(4, 5, d_model)
        
        output = moe(x)
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())
        print("MoE Forward: OK")
        
    def test_tp_linear_instantiation(self):
        print("\nTesting Tensor Parallel Linear Layer...")
        # Just check it initializes without error in single-process mode
        layer = ColumnParallelLinear(32, 64)
        x = torch.randn(4, 32)
        out = layer(x)
        # In single process (world_size=1), output should be full size
        self.assertEqual(out.shape, (4, 64))
        print("TP Linear: OK")

if __name__ == '__main__':
    unittest.main()
