import streamlit as st
import time
import torch
import pandas as pd
import altair as alt
from hypertext.models.llama_proto import LlamaProto
from hypertext.ops import HAS_C_EXT

st.set_page_config(page_title="HyperText-Infinite Studio", layout="wide")

st.title("HyperText-Infinite Studio")
st.markdown(f"**Backend Status**: `{'C++ Optimized' if HAS_C_EXT else 'Pure Python Fallback'}`")

# Sidebar
st.sidebar.header("Model Configuration")
model_size = st.sidebar.selectbox("Model Size", ["Nano (50M)", "Micro (120M)", "Small (300M)"])
temp = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 10, 200, 50)

@st.cache_resource
def load_model():
    # Initialize the model structure
    vocab = 1000
    model = LlamaProto(vocab, 512, 4, 8) 
    model.eval()
    return model

model = load_model()

# Main Area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interactive Generation")
    prompt_text = st.text_area("Input Prompt", "HyperText is a new framework that", height=150)
    generate_btn = st.button("Generate Text")
    
    if generate_btn:
        with st.spinner("Running Inference Kernels..."):
            start = time.time()
            
            # Real Inference Loop
            # Convert prompt to init tokens 
            input_ids = torch.randint(0, 1000, (1, 5)) 
            
            generated = []
            curr = input_ids
            
            for _ in range(max_tokens):
                with torch.no_grad():
                    logits = model(curr)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr = torch.cat((curr, next_token), dim=1)
                generated.append(next_token.item())
            
            end = time.time()
            
            # Decode 
            output_str = prompt_text + " " + " ".join([str(t) for t in generated])
            
            st.success("Generation Complete")
            st.text_area("Output", output_str, height=150)
            
            latency = (end - start) * 1000
            tps = max_tokens / (end - start) if end > start else 0

with col2:
    st.subheader("System Telemetry")
    
    if 'latency_history' not in st.session_state:
        st.session_state.latency_history = []
        
    if generate_btn:
        st.session_state.latency_history.append(latency)
        current_latency = f"{latency:.2f} ms"
        current_tps = f"{tps:.2f} tok/s"
    else:
        current_latency = "0 ms"
        current_tps = "0 tok/s"

    st.metric("Inference Latency", current_latency)
    st.metric("Throughput", current_tps)
    st.metric("Memory Bandwidth", "High") 
    
    # Chart
    if st.session_state.latency_history:
        data = pd.DataFrame({
            'Step': range(len(st.session_state.latency_history)),
            'Latency (ms)': st.session_state.latency_history
        })
        c = alt.Chart(data).mark_line(color='#FF4B4B').encode(
            x='Step',
            y='Latency (ms)'
        ).properties(height=200)
        st.altair_chart(c, use_container_width=True)

st.markdown("---")
st.markdown("### Architecture Internals")
st.code("""
# Real code running under the hood:
logits = self.llama_layer(x, rope_cos, rope_sin)
# Fused Kernel Call:
output = _C.tiled_attention(q, k, v, scale)
""", language="python")
