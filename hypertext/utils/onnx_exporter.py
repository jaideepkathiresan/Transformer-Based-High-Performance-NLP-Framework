import torch
import torch.onnx
from hypertext.models.llama_proto import LlamaProto

def export_to_onnx(model, output_path="llama.onnx"):
    print(f"Exporting model to {output_path}...")
    
    # Dummy input for tracing
    vocab_size = model.embed.num_embeddings
    dummy_input = torch.randint(0, vocab_size, (1, 32)) # (Batch, Seq)
    
    # Set to eval mode
    model.eval()
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print("Export successful.")
    
if __name__ == "__main__":
    # Demo export
    model = LlamaProto(vocab_size=1000, d_model=256, n_layers=2, n_heads=4)
    export_to_onnx(model)
