FROM nvcr.io/nvidia/vllm:25.12.post1-py3

WORKDIR /workspace

# Install system dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git curl wget sudo \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements_docker.txt /workspace/requirements_docker.txt
RUN pip install -r requirements_docker.txt