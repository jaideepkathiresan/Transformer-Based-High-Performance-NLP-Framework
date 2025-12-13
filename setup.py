from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import sys

# High-Performance C++ Extension Build
# Set ENABLE_C_EXT=1 env var to force build, otherwise it tries and falls back
ENABLE_C_EXT = os.getenv('ENABLE_C_EXT', '0') == '1'
FORCE_NO_C_EXT = os.getenv('FORCE_NO_C_EXT', '0') == '1'

csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc')

# Define extensions
sources = [
    os.path.join(csrc_dir, 'ops.cpp'),
    os.path.join(csrc_dir, 'feed_forward.cpp'),
    os.path.join(csrc_dir, 'attention.cpp'),
    os.path.join(csrc_dir, 'rms_norm.cpp'),
    os.path.join(csrc_dir, 'rope.cpp'),
]

ext_modules = [
    CppExtension(
        name='hypertext._C',
        sources=sources,
        extra_compile_args=['-O3', '-fopenmp'] if os.name != 'nt' else ['/O2', '/openmp'],
    )
]

def run_setup(with_ext):
    if with_ext:
        print("Building with High-Performance C++ Extensions...")
        cmd = {'build_ext': BuildExtension}
        ext = ext_modules
    else:
        print("Building in Pure Python Mode (No C++ Extensions)...")
        cmd = {}
        ext = []

    setup(
        name='hypertext',
        version='0.2.0', # Version bump for 10x scale
        description='HyperText: Enterprise-Grade NLP Framework',
        packages=find_packages(),
        ext_modules=ext,
        cmdclass=cmd,
        install_requires=['torch', 'transformers'],
    )

if __name__ == "__main__":
    # Auto-detection logic
    try:
        # If explicitly requested or default, try building ext
        if not FORCE_NO_C_EXT:
            run_setup(True)
        else:
            run_setup(False)
    except Exception as e:
        print(f"\nExample Compilation Failed: {e}")
        print("Falling back to Pure Python mode.")
        run_setup(False)
