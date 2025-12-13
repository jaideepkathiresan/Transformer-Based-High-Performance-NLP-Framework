import argparse
import sys
import subprocess
import os

def run_script(path):
    # We assume CWD is project root
    cmd = [sys.executable, path]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser(prog="python -m hypertext", description="HyperText-Infinite CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("pretrain", help="Run LLaMA Pretraining Simulation")
    subparsers.add_parser("generate", help="Run Inference Generation Demo")
    subparsers.add_parser("demo", help="Run Architecture Scale Demo")
    subparsers.add_parser("benchmark", help="Run Speed Benchmarks")
    subparsers.add_parser("serve-bench", help="Run Serving Load Test (requires server running)")
    subparsers.add_parser("dashboard", help="Launch Streamlit Dashboard")
    
    args = parser.parse_args()
    
    if args.command == "pretrain":
        run_script("examples/pretrain.py")
        
    elif args.command == "generate":
        run_script("examples/generate_demo.py")
        
    elif args.command == "demo":
        run_script("examples/scale_demo.py")
        
    elif args.command == "benchmark":
        run_script("benchmarks/benchmark_suite.py")

    elif args.command == "serve-bench":
        run_script("scripts/benchmark_serving.py")
        
    elif args.command == "dashboard":
        print("Launching Streamlit...")
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "serving/dashboard.py"])
        
    else:
        parser.print_help()
        
if __name__ == "__main__":
    main()
