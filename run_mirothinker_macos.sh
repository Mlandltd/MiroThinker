#!/usr/bin/env bash
# run_mirothinker_macos.sh
# End-to-end setup to run MiroThinker locally on an Apple Silicon MacBook Pro M3.
# Optimized for MacBook Pro M3 with 128GB RAM and 40-core GPU.
# Uses MiroThinker-v1.0-72B-Q4_K_S for best performance on this hardware.

set -euo pipefail

########################################
# 0. Basic checks / Homebrew
########################################

if [[ "$OSTYPE" != "darwin"* ]]; then
  echo "This script is intended for macOS (Apple Silicon)."
  exit 1
fi

# Detect M-series chip
CHIP_TYPE=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
echo "Detected: $CHIP_TYPE"

# Install Homebrew if missing
if ! command -v brew &>/dev/null; then
  echo "[0/9] Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "[0/9] Installing system dependencies via Homebrew..."
brew update
brew install git python3 cmake pkg-config || true

########################################
# 1. Python venv + uv
########################################

echo "[1/9] Creating Python virtual environment..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/mirothinker_env"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[1/9] Installing uv and huggingface_hub in the venv..."
pip install --upgrade pip
pip install uv huggingface_hub

########################################
# 2. Setup MiroThinker repo (use current directory)
########################################

echo "[2/9] Setting up MiroThinker repo..."
cd "$SCRIPT_DIR"

# If repo doesn't exist, clone it
if [ ! -d "apps/miroflow-agent" ]; then
  echo "[2/9] MiroThinker repo not found in current directory."
  echo "      Please ensure you're running this script from the MiroThinker root directory."
  echo "      Or the repo will be cloned to: $SCRIPT_DIR"
  if [ ! -d ".git" ]; then
    echo "[2/9] Cloning MiroThinker repo..."
    git clone https://github.com/MiroMindAI/MiroThinker.git temp_mirothinker
    # Copy structure if needed - for now assume we're in the repo
  fi
fi

cd "$SCRIPT_DIR/apps/miroflow-agent" || {
  echo "ERROR: Could not find apps/miroflow-agent directory."
  echo "       Please run this script from the MiroThinker root directory."
  exit 1
}

echo "[2/9] Installing Python dependencies with uv..."
uv sync

########################################
# 3. Download 72B GGUF model (optimized for 128GB RAM)
########################################

echo "[3/9] Preparing model directory..."
MODEL_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODEL_DIR"

# For MacBook Pro M3 with 128GB RAM, we'll try to get the best available model
# Note: GGUF quantized versions may need to be downloaded manually from community sources
# The official repos have full models which can be very large

# Default to 30B model which is more likely to have GGUF versions available
# Note: Actual filenames in bartowski repo are prefixed with "miromind-ai_"
MODEL_NAME="MiroThinker-v1.0-30B-Q5_K_M.gguf"
MODEL_NAME_ACTUAL="miromind-ai_MiroThinker-v1.0-30B-Q5_K_M.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
MODEL_PATH_ACTUAL="$MODEL_DIR/$MODEL_NAME_ACTUAL"

echo "[3/9] Model selection for MacBook Pro M3 (128GB RAM, 40-core GPU):"
echo "      Default: $MODEL_NAME (~18GB) - Good balance of quality and speed"
echo "      Alternative options:"
echo "        - MiroThinker-v1.0-72B-Q4_K_S.gguf (~40-45GB) - Best quality if available"
echo "        - MiroThinker-v1.0-30B-Q5_K_M.gguf (~20GB) - Higher quality 30B"
echo "        - MiroThinker-v1.0-8B-Q4_K_M.gguf (~5GB) - Fastest, smaller"

if [ ! -f "$MODEL_PATH" ]; then
  echo "[3/9] Logging into Hugging Face for model download..."
  echo "      If prompted, paste your HF token from https://huggingface.co/settings/tokens"
  if ! huggingface-cli whoami &>/dev/null; then
    huggingface-cli login
  else
    echo "[3/9] Already logged into Hugging Face."
  fi

  echo "[3/9] Downloading $MODEL_NAME..."
  echo "      Note: GGUF quantized models may need to be downloaded manually"
  echo "      from community sources as they're not always in official repos."
  
  DOWNLOAD_SUCCESS=false
  
  # Try to download 30B Q5_K_M from bartowski (recommended for 128GB RAM)
  echo "      Trying: bartowski/miromind-ai_MiroThinker-v1.0-30B-GGUF"
  echo "      Downloading: $MODEL_NAME_ACTUAL"
  if hf download bartowski/miromind-ai_MiroThinker-v1.0-30B-GGUF \
    "$MODEL_NAME_ACTUAL" \
    --local-dir "$MODEL_DIR" 2>&1; then
    sleep 2
    # Check for actual filename or rename if needed
    if [ -f "$MODEL_PATH_ACTUAL" ]; then
      # Rename to expected name for consistency
      mv "$MODEL_PATH_ACTUAL" "$MODEL_PATH" 2>/dev/null || MODEL_PATH="$MODEL_PATH_ACTUAL"
      DOWNLOAD_SUCCESS=true
      echo "[3/9] ✅ Successfully downloaded $MODEL_NAME"
    elif [ -f "$MODEL_PATH" ]; then
      DOWNLOAD_SUCCESS=true
      echo "[3/9] ✅ Successfully downloaded $MODEL_NAME"
    fi
  fi
  
  # If that failed, provide clear manual download instructions
  if [ "$DOWNLOAD_SUCCESS" = false ]; then
    echo
    echo "[3/9] ⚠️  Automatic download from bartowski failed."
    echo "      GGUF quantized models need to be downloaded manually."
    echo
    echo "      📥 MANUAL DOWNLOAD INSTRUCTIONS:"
    echo "      1. Visit Hugging Face and search for quantized MiroThinker models:"
    echo "         https://huggingface.co/models?search=mirothinker+gguf"
    echo
    echo "      2. Look for repositories like:"
    echo "         - bartowski/miromind-ai_MiroThinker-v1.0-30B-GGUF"
    echo "         - TheBloke/MiroThinker-v1.0-30B-GGUF"
    echo "         - Or search for 'MiroThinker GGUF'"
    echo
    echo "      3. Download the model file: $MODEL_NAME"
    echo "         (or any Q4_K_S, Q5_K_M, or Q6_K_L variant)"
    echo
    echo "      4. Place the downloaded .gguf file in:"
    echo "         $MODEL_DIR"
    echo
    echo "      5. If the filename is different, rename it to: $MODEL_NAME"
    echo "         OR update MODEL_NAME in this script to match your file"
    echo
    echo "      💡 RECOMMENDED FOR YOUR 128GB RAM:"
    echo "         - MiroThinker-v1.0-30B-Q5_K_M.gguf (~20GB) - Great quality"
    echo "         - MiroThinker-v1.0-30B-Q4_K_S.gguf (~18GB) - Good balance"
    echo "         - MiroThinker-v1.0-72B-Q4_K_S.gguf (~40GB) - Best if available"
    echo
    read -p "      Press Enter after you've downloaded the model to $MODEL_DIR, or Ctrl+C to exit: "
    
    # Check if file exists now (try both naming conventions)
    if [ -f "$MODEL_PATH" ] || [ -f "$MODEL_PATH_ACTUAL" ]; then
      if [ -f "$MODEL_PATH_ACTUAL" ] && [ ! -f "$MODEL_PATH" ]; then
        # Use actual filename if it exists
        MODEL_PATH="$MODEL_PATH_ACTUAL"
        MODEL_NAME="$MODEL_NAME_ACTUAL"
        echo "[3/9] Found model with actual filename: $MODEL_NAME"
      fi
      DOWNLOAD_SUCCESS=true
    else
      # Check if any .gguf file exists in the directory
      GGUF_FILES=$(find "$MODEL_DIR" -name "*.gguf" 2>/dev/null | head -1)
      if [ -n "$GGUF_FILES" ]; then
        echo "[3/9] Found GGUF file: $GGUF_FILES"
        echo "      Using this file..."
        MODEL_NAME=$(basename "$GGUF_FILES")
        MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
        echo "[3/9] Using: $MODEL_NAME"
        DOWNLOAD_SUCCESS=true
      else
        echo
        echo "ERROR: No .gguf model file found in $MODEL_DIR"
        echo "       Please download a MiroThinker GGUF model and place it there."
        echo
        echo "       Quick download command:"
        echo "       cd /Users/admin/Documents/AI/MiroThinker"
        echo "       source mirothinker_env/bin/activate"
        echo "       hf download bartowski/miromind-ai_MiroThinker-v1.0-30B-GGUF \\"
        echo "         \"miromind-ai_MiroThinker-v1.0-30B-Q5_K_M.gguf\" --local-dir models"
        exit 1
      fi
    fi
  fi
else
  echo "[3/9] Model already present at $MODEL_PATH, skipping download."
fi

echo "[3/9] Model ready: $MODEL_PATH"

########################################
# 4. Build llama.cpp with Metal
########################################

echo "[4/9] Cloning and building llama.cpp with Metal support..."
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"

if [ ! -d "$LLAMA_DIR" ]; then
  echo "[4/9] Cloning llama.cpp..."
  cd "$SCRIPT_DIR"
  git clone https://github.com/ggerganov/llama.cpp.git
fi

cd "$LLAMA_DIR"

mkdir -p build
cd build

echo "[4/9] Configuring CMake with Metal support..."
cmake .. -DLLAMA_METAL=ON -DLLAMA_VULKAN=OFF -DLLAMA_CUBLAS=OFF -DCMAKE_BUILD_TYPE=Release

echo "[4/9] Building llama.cpp (this may take several minutes)..."
make -j"$(sysctl -n hw.ncpu)"

# The server binary is in the bin directory and named llama-server
SERVER_BIN="$PWD/bin/llama-server"
if [ ! -x "$SERVER_BIN" ]; then
  # Try alternative location (older builds)
  if [ -x "$PWD/server" ]; then
    SERVER_BIN="$PWD/server"
  else
    echo "ERROR: llama.cpp server binary not found."
    echo "       Tried: $PWD/bin/llama-server"
    echo "       Tried: $PWD/server"
    echo "       Please check the build output for errors."
    exit 1
  fi
fi

echo "[4/9] llama.cpp built successfully."
echo "      Server binary found at: $SERVER_BIN"

########################################
# 5. Start llama.cpp server
########################################

echo "[5/9] Starting llama.cpp server (OpenAI-style API)..."
LLM_PORT=8000
# MiroThinker v1.0 supports up to 256K context
LLM_CTX=262144
# With 128GB unified memory, offload all layers to Metal GPU for best performance
LLM_N_GPU_LAYERS=99
# Use all CPU cores for processing
CPU_THREADS=$(sysctl -n hw.ncpu)

echo "[5/9] Server configuration:"
echo "      Port: $LLM_PORT"
echo "      Context size: $LLM_CTX (256K - full v1.0 support)"
echo "      GPU layers: $LLM_N_GPU_LAYERS (all layers on Metal GPU)"
echo "      CPU threads: $CPU_THREADS"

# Kill any existing server on the same port
if pgrep -f "server .*--port $LLM_PORT" >/dev/null 2>&1; then
  echo "[5/9] Killing existing llama.cpp server on port $LLM_PORT..."
  pkill -f "server .*--port $LLM_PORT" || true
  sleep 2
fi

echo "[5/9] Starting server in background..."
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
echo "[5/9] llama.cpp server started with PID $SERVER_PID on port $LLM_PORT."
echo "      Logs: /tmp/mirothinker_llama_server.log"
echo "      Waiting for server to initialize..."
sleep 10

# Check if server is running
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "ERROR: Server process died. Check logs: /tmp/mirothinker_llama_server.log"
  tail -20 /tmp/mirothinker_llama_server.log
  exit 1
fi

echo "[5/9] Server is running. You can monitor it with:"
echo "      tail -f /tmp/mirothinker_llama_server.log"

########################################
# 6. Setup .env for tools (user must create manually)
########################################

echo "[6/9] Setting up .env configuration..."
cd "$SCRIPT_DIR/apps/miroflow-agent"

if [ ! -f .env ]; then
  # Check if .env.example exists and copy it
  if [ -f .env.example ]; then
    echo "[6/9] Found .env.example, copying to .env..."
    cp .env.example .env
    echo "[6/9] .env created from .env.example template."
  else
    echo "[6/9] .env file not found and .env.example not available."
    echo "      You need to create .env manually before running the agent."
  fi
else
  echo "[6/9] .env file already exists, skipping creation."
fi

echo
echo "[6/9] IMPORTANT: Configure your API keys in apps/miroflow-agent/.env"
echo "      Required keys for MiroThinker v1.0 (single_agent_keep5):"
echo "        - SERPER_API_KEY (get from https://serper.dev/)"
echo "        - SERPER_BASE_URL=\"https://google.serper.dev\""
echo "        - JINA_API_KEY (get from https://jina.ai/)"
echo "        - JINA_BASE_URL=\"https://r.jina.ai\""
echo "        - E2B_API_KEY (get from https://e2b.dev/)"
echo "        - SUMMARY_LLM_BASE_URL (e.g., \"https://api.openai.com/v1\")"
echo "        - SUMMARY_LLM_MODEL_NAME (e.g., \"gpt-4o-mini\")"
echo "        - SUMMARY_LLM_API_KEY"
echo "        - OPENAI_API_KEY (optional, only for benchmark evaluation)"
echo
echo "      If .env doesn't exist, create it manually with your API keys."
echo "      The script will NOT store or hardcode any API keys."
echo

########################################
# 7. Verify server is ready
########################################

echo "[7/9] Verifying llama.cpp server is ready..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  if curl -s "http://127.0.0.1:$LLM_PORT/health" >/dev/null 2>&1 || \
     curl -s "http://127.0.0.1:$LLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "[7/9] Server is ready!"
    break
  fi
  RETRY_COUNT=$((RETRY_COUNT + 1))
  echo "[7/9] Waiting for server... ($RETRY_COUNT/$MAX_RETRIES)"
  sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo "WARNING: Server may not be fully ready, but continuing..."
fi

########################################
# 8. Run the MiroThinker agent
########################################

echo "[8/9] Launching the MiroThinker agent with local LLM and tools..."
cd "$SCRIPT_DIR/apps/miroflow-agent"

# Export .env vars into current shell
set -a
# shellcheck disable=SC1091
[ -f .env ] && source .env
set +a

echo "[8/9] Starting MiroThinker agent..."
echo "      Model: MiroThinker-v1.0-72B-Q4_K_S"
echo "      Agent config: single_agent_keep5 (recommended for v1.0)"
echo "      Base URL: http://127.0.0.1:${LLM_PORT}/v1"
echo

uv run main.py \
  llm=qwen-3 \
  agent=single_agent_keep5 \
  llm.base_url="http://127.0.0.1:${LLM_PORT}/v1"

########################################
# 9. Cleanup hint
########################################

echo
echo "[9/9] MiroThinker agent has exited."
echo
echo "To stop the llama.cpp server, run:"
echo "  pkill -f \"server .*--port $LLM_PORT\" || true"
echo
echo "To restart just the agent (if server is still running):"
echo "  cd $SCRIPT_DIR/apps/miroflow-agent"
echo "  source $VENV_DIR/bin/activate"
echo "  uv run main.py llm=qwen-3 agent=single_agent_keep5 llm.base_url=\"http://127.0.0.1:${LLM_PORT}/v1\""
echo
echo "Done."