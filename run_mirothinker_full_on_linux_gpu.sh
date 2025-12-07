#!/usr/bin/env bash
# run_mirothinker_full_on_linux_gpu.sh
# End-to-end setup to run MiroThinker locally with tools on a single 24 GB GPU (RTX 5090).
# Optimized for RTX 5090 with 24GB VRAM.
# - Installs dependencies (Linux / Ubuntu-like)
# - Sets up Python venv + uv
# - Sets up MiroThinker repo (uses current directory)
# - Downloads MiroThinker‑v1.0‑30B GGUF (Q4_K_S or Q5_K_M)
# - Builds llama.cpp with CUDA and runs it as an OpenAI-style API server
# - Configures .env for tool API keys (user must add keys manually)
# - Runs the MiroThinker agent with tool usage enabled

set -euo pipefail

#############################
# 0. Basic sanity checks
#############################

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
  echo "This script is written for Linux (Debian/Ubuntu-like)."
  echo "For macOS, use run_mirothinker_macos.sh"
  exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers before continuing."
  exit 1
fi

# Check GPU info
echo "Detected GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

#############################
# 1. System dependencies
#############################

echo "[1/9] Installing system dependencies via apt (sudo required)..."
sudo apt update
sudo apt install -y \
  git \
  python3 python3-venv python3-pip \
  make cmake build-essential \
  wget \
  || echo "Warning: Some packages may already be installed"

#############################
# 2. Python environment + uv
#############################

echo "[2/9] Creating Python virtual environment..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/mirothinker_env"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/9] Installing uv and huggingface_hub in venv..."
pip install --upgrade pip
pip install uv huggingface_hub

#############################
# 3. Setup MiroThinker repo (use current directory)
#############################

echo "[3/9] Setting up MiroThinker repo..."
cd "$SCRIPT_DIR"

# If repo doesn't exist, clone it
if [ ! -d "apps/miroflow-agent" ]; then
  echo "[3/9] MiroThinker repo not found in current directory."
  echo "      Please ensure you're running this script from the MiroThinker root directory."
  if [ ! -d ".git" ]; then
    echo "[3/9] Cloning MiroThinker repo..."
    git clone https://github.com/MiroMindAI/MiroThinker.git temp_mirothinker
    # Copy structure if needed - for now assume we're in the repo
  fi
fi

cd "$SCRIPT_DIR/apps/miroflow-agent" || {
  echo "ERROR: Could not find apps/miroflow-agent directory."
  echo "       Please run this script from the MiroThinker root directory."
  exit 1
}

echo "[3/9] Installing Python dependencies with uv..."
uv sync

#############################
# 4. Download 30B GGUF model (optimized for 24GB VRAM)
#############################

echo "[4/9] Preparing model directory..."
MODEL_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODEL_DIR"

# For RTX 5090 with 24GB VRAM, use 30B model with Q4_K_S or Q5_K_M quantization
# Note: Actual filenames in bartowski repo are prefixed with "miromind-ai_"
MODEL_NAME="MiroThinker-v1.0-30B-Q4_K_S.gguf"
MODEL_NAME_ACTUAL="miromind-ai_MiroThinker-v1.0-30B-Q4_K_S.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
MODEL_PATH_ACTUAL="$MODEL_DIR/$MODEL_NAME_ACTUAL"

echo "[4/9] Model selection for RTX 5090 (24GB VRAM):"
echo "      Default: $MODEL_NAME (~18GB) - Good balance, fits comfortably"
echo "      Alternative options:"
echo "        - MiroThinker-v1.0-30B-Q5_K_M.gguf (~20GB) - Higher quality"
echo "        - MiroThinker-v1.0-30B-Q3_K_XL.gguf (~15GB) - Lighter, faster"

if [ ! -f "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH_ACTUAL" ]; then
  echo "[4/9] Logging into Hugging Face for model download..."
  if ! huggingface-cli whoami &>/dev/null; then
    echo "      If prompted, paste your HF token from https://huggingface.co/settings/tokens"
    huggingface-cli login
  else
    echo "[4/9] Already logged into Hugging Face."
  fi

  echo "[4/9] Downloading $MODEL_NAME_ACTUAL (this may take a while, ~18GB)..."
  echo "      Repository: bartowski/miromind-ai_MiroThinker-v1.0-30B-GGUF"
  
  DOWNLOAD_SUCCESS=false
  
  # Try to download using hf command (newer)
  if hf download bartowski/miromind-ai_MiroThinker-v1.0-30B-GGUF \
    "$MODEL_NAME_ACTUAL" \
    --local-dir "$MODEL_DIR" 2>&1; then
    sleep 2
    if [ -f "$MODEL_PATH_ACTUAL" ]; then
      # Rename to expected name for consistency
      mv "$MODEL_PATH_ACTUAL" "$MODEL_PATH" 2>/dev/null || MODEL_PATH="$MODEL_PATH_ACTUAL"
      DOWNLOAD_SUCCESS=true
      echo "[4/9] ✅ Successfully downloaded $MODEL_NAME"
    elif [ -f "$MODEL_PATH" ]; then
      DOWNLOAD_SUCCESS=true
      echo "[4/9] ✅ Successfully downloaded $MODEL_NAME"
    fi
  fi
  
  if [ "$DOWNLOAD_SUCCESS" = false ]; then
    echo
    echo "[4/9] ⚠️  Automatic download failed."
    echo "      Please download manually:"
    echo "      1. Visit: https://huggingface.co/models?search=mirothinker+gguf"
    echo "      2. Download: $MODEL_NAME_ACTUAL (or Q4_K_S variant)"
    echo "      3. Place in: $MODEL_DIR"
    echo
    read -p "      Press Enter after downloading, or Ctrl+C to exit: "
    
    # Check if file exists now
    if [ -f "$MODEL_PATH_ACTUAL" ]; then
      MODEL_PATH="$MODEL_PATH_ACTUAL"
      MODEL_NAME="$MODEL_NAME_ACTUAL"
      echo "[4/9] Found model: $MODEL_NAME"
      DOWNLOAD_SUCCESS=true
    elif [ -f "$MODEL_PATH" ]; then
      DOWNLOAD_SUCCESS=true
    else
      GGUF_FILES=$(find "$MODEL_DIR" -name "*.gguf" 2>/dev/null | head -1)
      if [ -n "$GGUF_FILES" ]; then
        MODEL_NAME=$(basename "$GGUF_FILES")
        MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
        echo "[4/9] Using found model: $MODEL_NAME"
        DOWNLOAD_SUCCESS=true
      fi
    fi
    
    if [ "$DOWNLOAD_SUCCESS" = false ]; then
      echo "ERROR: No .gguf model file found in $MODEL_DIR"
      exit 1
    fi
  fi
else
  if [ -f "$MODEL_PATH_ACTUAL" ]; then
    MODEL_PATH="$MODEL_PATH_ACTUAL"
    MODEL_NAME="$MODEL_NAME_ACTUAL"
  fi
  echo "[4/9] Model already present at $MODEL_PATH, skipping download."
fi

echo "[4/9] Model ready: $MODEL_PATH"

#############################
# 5. Build llama.cpp with CUDA
#############################

echo "[5/9] Cloning and building llama.cpp with CUDA support..."
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"

if [ ! -d "$LLAMA_DIR" ]; then
  echo "[5/9] Cloning llama.cpp..."
  cd "$SCRIPT_DIR"
  git clone https://github.com/ggerganov/llama.cpp.git
fi

cd "$LLAMA_DIR"

echo "[5/9] Building llama.cpp with CUDA..."
make clean

if command -v nvcc &>/dev/null; then
  echo "[5/9] CUDA detected (nvcc found). Building with LLAMA_CUBLAS=1..."
  make LLAMA_CUBLAS=1 -j"$(nproc)"
else
  echo "[5/9] ⚠️  CUDA toolkit not detected (nvcc not found)."
  echo "      Building CPU-only (will be very slow)."
  echo "      Please install CUDA toolkit for GPU acceleration."
  make -j"$(nproc)"
fi

# The server binary location depends on build method
# Try multiple possible locations
if [ -x "$LLAMA_DIR/build/bin/llama-server" ]; then
  SERVER_BIN="$LLAMA_DIR/build/bin/llama-server"
elif [ -x "$LLAMA_DIR/server" ]; then
  SERVER_BIN="$LLAMA_DIR/server"
elif [ -x "$LLAMA_DIR/build/server" ]; then
  SERVER_BIN="$LLAMA_DIR/build/server"
else
  echo "ERROR: llama.cpp server binary not found."
  echo "       Tried: $LLAMA_DIR/build/bin/llama-server"
  echo "       Tried: $LLAMA_DIR/server"
  echo "       Tried: $LLAMA_DIR/build/server"
  echo "       Please check the build output for errors."
  exit 1
fi

echo "[5/9] llama.cpp built successfully."
echo "      Server binary found at: $SERVER_BIN"

#############################
# 6. Start llama.cpp server
#############################

echo "[6/9] Starting llama.cpp server (OpenAI-style API)..."
LLM_PORT=8000
# MiroThinker v1.0 supports up to 256K context, but 128K is more practical for 24GB VRAM
LLM_CTX=131072
# For 24GB VRAM with 30B Q4_K_S model, use ~20 GPU layers (leaves room for KV cache)
LLM_N_GPU_LAYERS=20
CPU_THREADS=$(nproc)

echo "[6/9] Server configuration:"
echo "      Port: $LLM_PORT"
echo "      Context size: $LLM_CTX (128K - practical for 24GB VRAM)"
echo "      GPU layers: $LLM_N_GPU_LAYERS (optimized for 24GB VRAM)"
echo "      CPU threads: $CPU_THREADS"
echo "      Note: If you get OOM errors, reduce --n-gpu-layers to 16 or 12"

# Kill any existing server on that port
if pgrep -f "llama-server.*--port $LLM_PORT\|server .*--port $LLM_PORT" >/dev/null 2>&1; then
  echo "[6/9] Killing existing llama.cpp server on port $LLM_PORT..."
  pkill -f "llama-server.*--port $LLM_PORT\|server .*--port $LLM_PORT" || true
  sleep 2
fi

echo "[6/9] Starting server in background..."
"$SERVER_BIN" \
  -m "$MODEL_PATH" \
  --port "$LLM_PORT" \
  --ctx-size "$LLM_CTX" \
  --n-gpu-layers "$LLM_N_GPU_LAYERS" \
  --threads "$CPU_THREADS" \
  --batch-size 512 \
  --ubatch-size 512 \
  >/tmp/mirothinker_llama_server.log 2>&1 &

SERVER_PID=$!
echo "[6/9] llama.cpp server started with PID $SERVER_PID on port $LLM_PORT."
echo "      Logs: /tmp/mirothinker_llama_server.log"
echo "      Waiting for server to initialize..."
sleep 10

# Check if server is running
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "ERROR: Server process died. Check logs: /tmp/mirothinker_llama_server.log"
  tail -20 /tmp/mirothinker_llama_server.log
  exit 1
fi

echo "[6/9] Server is running. You can monitor it with:"
echo "      tail -f /tmp/mirothinker_llama_server.log"

#############################
# 7. Setup .env for tools (user must create manually)
#############################

echo "[7/9] Setting up .env configuration..."
cd "$SCRIPT_DIR/apps/miroflow-agent"

if [ ! -f .env ]; then
  # Check if .env.example exists and copy it
  if [ -f .env.example ]; then
    echo "[7/9] Found .env.example, copying to .env..."
    cp .env.example .env
    echo "[7/9] .env created from .env.example template."
  else
    echo "[7/9] .env file not found and .env.example not available."
    echo "      You need to create .env manually before running the agent."
  fi
else
  echo "[7/9] .env file already exists, skipping creation."
fi

echo
echo "[7/9] IMPORTANT: Configure your API keys in apps/miroflow-agent/.env"
echo "      Required keys for MiroThinker v1.0 (single_agent_keep5):"
echo "        - SERPER_API_KEY (get from https://serper.dev/)"
echo "        - SERPER_BASE_URL=\"https://google.serper.dev\""
echo "        - JINA_API_KEY (get from https://jina.ai/)"
echo "        - JINA_BASE_URL=\"https://r.jina.ai\""
echo "        - E2B_API_KEY (get from https://e2b.dev/)"
echo "        - SUMMARY_LLM_BASE_URL=\"https://api.openai.com/v1/chat/completions\""
echo "        - SUMMARY_LLM_MODEL_NAME=\"gpt-4o-mini\""
echo "        - SUMMARY_LLM_API_KEY"
echo "        - OPENAI_API_KEY (optional, only for benchmark evaluation)"
echo
echo "      If .env doesn't exist, create it manually with your API keys."
echo "      The script will NOT store or hardcode any API keys."
echo

#############################
# 8. Verify server is ready
#############################

echo "[8/9] Verifying llama.cpp server is ready..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  if curl -s "http://127.0.0.1:$LLM_PORT/health" >/dev/null 2>&1 || \
     curl -s "http://127.0.0.1:$LLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "[8/9] Server is ready!"
    break
  fi
  RETRY_COUNT=$((RETRY_COUNT + 1))
  echo "[8/9] Waiting for server... ($RETRY_COUNT/$MAX_RETRIES)"
  sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo "WARNING: Server may not be fully ready, but continuing..."
  echo "         Check logs: tail -f /tmp/mirothinker_llama_server.log"
fi

#############################
# 9. Run MiroThinker agent
#############################

echo "[9/9] Launching the MiroThinker agent with local LLM and tools..."
cd "$SCRIPT_DIR/apps/miroflow-agent"

# Export .env vars into current shell
set -a
# shellcheck disable=SC1091
[ -f .env ] && source .env
set +a

echo "[9/9] Starting MiroThinker agent..."
echo "      Model: $MODEL_NAME"
echo "      Agent config: single_agent_keep5 (recommended for v1.0)"
echo "      Base URL: http://127.0.0.1:${LLM_PORT}/v1"
echo

uv run main.py \
  llm=qwen-3 \
  agent=single_agent_keep5 \
  llm.base_url="http://127.0.0.1:${LLM_PORT}/v1"

########################################
# Cleanup hint
########################################

echo
echo "[9/9] MiroThinker agent has exited."
echo
echo "To stop the llama.cpp server, run:"
echo "  pkill -f \"llama-server.*--port $LLM_PORT\|server .*--port $LLM_PORT\" || true"
echo
echo "To restart just the agent (if server is still running):"
echo "  cd $SCRIPT_DIR/apps/miroflow-agent"
echo "  source $VENV_DIR/bin/activate"
echo "  uv run main.py llm=qwen-3 agent=single_agent_keep5 llm.base_url=\"http://127.0.0.1:${LLM_PORT}/v1\""
echo
echo "Done."