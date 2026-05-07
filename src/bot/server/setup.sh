#!/bin/bash

# 1. Download Anaconda
wget -c https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

# 2. Run installer (auto-confirm all prompts)
bash Anaconda3-2024.10-1-Linux-x86_64.sh -b

# 3. Load Conda into current shell

eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
# 4. Create and activate conda environment
conda create --name myenv python=3.12 -y
conda activate myenv

# 5. Install Python dependencies
pip install torch indic-num2words torchaudio rapidfuzz asyncio websockets webrtcvad torch numpy transformers noisereduce soundfile openai chromadb sentence-transformers elevenlabs

# 6. go to the directly
cd /workspace/
