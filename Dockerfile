# Stage 1: Build
FROM python:3.9-slim as builder

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install dependencies and build extension
# Force C++ extension build
ENV ENABLE_C_EXT=1
RUN pip install --user -e .

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Update PATH
ENV PATH=/root/.local/bin:$PATH

# Default command
CMD ["python", "examples/pretrain.py"]
