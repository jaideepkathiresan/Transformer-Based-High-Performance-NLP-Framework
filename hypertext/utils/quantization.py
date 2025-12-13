import torch

class Quantizer:
    @staticmethod
    def quantize_int8(tensor):
        """
        Simulate Int8 Quantization.
        Returns:
            quantized_tensor, scale, zero_point
        """
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / 255.0
        zero_point = (-min_val / scale).round().clamp(0, 255).byte()
        
        q_tensor = torch.clamp((tensor / scale) + zero_point, 0, 255).byte()
        return q_tensor, scale, zero_point
    
    @staticmethod
    def dequantize_int8(q_tensor, scale, zero_point):
        return (q_tensor.float() - zero_point.float()) * scale

class Linear8bit(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        weight = torch.randn(out_features, in_features)
        self.register_buffer('weight_idx', None) # Placeholder for quantized indices
        self.register_buffer('scale', None)
        self.register_buffer('zp', None)
        self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Initial quantization
        q_w, s, zp = Quantizer.quantize_int8(weight)
        self.weight_idx = q_w
        self.scale = s
        self.zp = zp
        
    def forward(self, x):
        # On-the-fly dequantize for calculation (Fake Quant)
        # Real kernels would do Int8 GEMM
        w = Quantizer.dequantize_int8(self.weight_idx, self.scale, self.zp)
        return torch.nn.functional.linear(x, w, self.bias)
