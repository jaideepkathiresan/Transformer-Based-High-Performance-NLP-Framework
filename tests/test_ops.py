import unittest
import torch
import torch.nn.functional as F
try:
    from hypertext.ops import fused_feed_forward
except ImportError:
    # If package not installed, we might fail or need to mock.
    # We assume 'pip install -e .' has been run or we are in the root.
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from hypertext.ops import fused_feed_forward

class TestFusedOps(unittest.TestCase):
    def test_fused_ffn_correctness(self):
        batch_size = 4
        input_dim = 8
        hidden_dim = 16
        output_dim = 8
        
        # Inputs
        x = torch.randn(batch_size, input_dim)
        w1 = torch.randn(hidden_dim, input_dim)
        b1 = torch.randn(hidden_dim)
        w2 = torch.randn(output_dim, hidden_dim)
        b2 = torch.randn(output_dim)
        
        # Standard PyTorch Reference
        # 1. Linear
        # x @ w1.t() + b1
        h = F.linear(x, w1, b1)
        # 2. ReLU
        h = F.relu(h)
        # 3. Linear
        expected = F.linear(h, w2, b2)
        
        # Custom Op
        # Our op signature: input, w1, b1, w2, b2
        actual = fused_feed_forward(x, w1, b1, w2, b2)
        
        # Check closeness
        # Floating point differences might differ slightly due to accumulated error order, 
        # but should be very close.
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        print("\nTest Passed: Custom Fused Op matches PyTorch Standard Op output.")

if __name__ == '__main__':
    unittest.main()
