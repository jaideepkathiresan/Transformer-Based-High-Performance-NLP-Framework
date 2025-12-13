import http.server
import socketserver
import json
import time
import torch
import torch.nn.functional as F
from hypertext.models.llama_proto import LlamaProto

# Global Model Cache
MODEL = None
VOCAB_SIZE = 1000  # Matching our demo config
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8

def load_model():
    global MODEL
    if MODEL is None:
        print("Loading Model into RAM...")
        # Initialize model structure
        MODEL = LlamaProto(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS)
        MODEL.eval()
        print("Model Loaded.")
    return MODEL

class InferenceHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data)
                prompt_ids = data.get('input_ids', [0])
                max_tokens = data.get('max_tokens', 20)
                
                # Convert to tensor
                model = load_model()
                ctx = torch.tensor([prompt_ids], dtype=torch.long)
                
                start = time.time()
                # Simple generation loop
                for _ in range(max_tokens):
                    with torch.no_grad():
                        logits = model(ctx)
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        ctx = torch.cat((ctx, next_token), dim=1)
                
                duration = time.time() - start
                
                response = {
                    "output_ids": ctx[0].tolist(),
                    "tokens_generated": max_tokens,
                    "latency_ms": duration * 1000
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8080):
    handler = InferenceHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"HyperText Inference Server serving at port {port}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
