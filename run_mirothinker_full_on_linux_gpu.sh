#!/usr/bin/env bash
# run_mirothinker_full.sh
# End-to-end setup to run MiroThinker locally with tools on a single 24 GB GPU.
# - Installs dependencies (Linux / Ubuntu-like)
# - Sets up Python venv + uv
# - Clones MiroThinker repo
# - Downloads MiroThinker‑v1.0‑30B GGUF (Q4_K_S)
# - Builds llama.cpp with CUDA and runs it as an OpenAI-style API server
# - Creates .env with placeholders for tool API keys
# - Runs the MiroThinker agent with tool usage enabled

set -euo pipefail

#############################
# 0. Basic sanity checks
#############################

if ["$OSTYPE" != "linux-gnu"*](# "Reference "$OSTYPE" != "linux-gnu"* ||| __GENERATING_DETAILS__"); then
  echo "This script is written for Linux (Debian/Ubuntu-like)."
  echo "For macOS/Windows, adapt the install/build commands manually."
  exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi not found. Please install NVIDIA drivers before continuing."
  exit 1
fi

#############################
# 1. System dependencies
#############################

echo "[1/8] Installing system dependencies via apt (sudo required)..."
sudo apt update
sudo apt install -y \
  git \
  python3 python3-venv python3-pip \
  make cmake build-essential \
  wget

#############################
# 2. Python environment + uv
#############################

echo "[2/8] Creating Python virtual environment..."
mkdir -p ~/mirothinker_env
python3 -m venv ~/mirothinker_env/venv
# shellcheck disable=SC1091
source ~/mirothinker_env/venv/bin/activate

echo "[2/8] Installing uv and huggingface_hub in venv..."
pip install --upgrade pip
pip install uv huggingface_hub

#############################
# 3. Clone MiroThinker repo
#############################

echo "[3/8] Cloning MiroThinker repo..."
cd ~
if [! -d MiroThinker](# "Reference ! -d MiroThinker \|\|\| __GENERATING_DETAILS__"); then
  git clone https://github.com/MiroMindAI/MiroThinker
fi
cd MiroThinker

echo "[3/8] Installing Python dependencies with uv..."
uv sync

#############################
# 4. Download 30B GGUF model
#############################

echo "[4/8] Preparing model directory..."
mkdir -p ~/MiroThinker/models
cd ~/MiroThinker

MODEL_NAME="MiroThinker-v1.0-30B-Q4_K_S.gguf"
MODEL_DIR="models"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

if [! -f "$MODEL_PATH"](# "Reference ! -f "$MODEL_PATH" ||| __GENERATING_DETAILS__"); then
  echo "[4/8] Logging into Hugging Face (one-time) for model download..."
  echo "If prompted, paste your HF token from https://huggingface.co/settings/tokens"
  huggingface-cli login

  echo "[4/8] Downloading $MODEL_NAME (this may take a while)..."
  huggingface-cli download bartowski/miromind-ai_MiroThinker-v1.0-30B-GGUF \
    --include "$MODEL_NAME" \
    --local-dir "$MODEL_DIR"
else
  echo "[4/8] Model already exists at $MODEL_PATH, skipping download."
fi

if [! -f "$MODEL_PATH"](# "Reference ! -f "$MODEL_PATH" ||| __GENERATING_DETAILS__"); then
  echo "ERROR: Model file $MODEL_PATH not found after download."
  exit 1
fi

#############################
# 5. Build llama.cpp + server
#############################

echo "[5/8] Cloning and building llama.cpp..."
cd ~
if [! -d llama.cpp](# "Reference ! -d llama.cpp \|\|\| __GENERATING_DETAILS__"); then
  git clone https://github.com/ggerganov/llama.cpp
fi
cd llama.cpp

make clean
if command -v nvcc &>/dev/null; then
  echo "[5/8] CUDA detected (nvcc found). Building with LLAMA_CUBLAS=1..."
  make LLAMA_CUBLAS=1 -j"$(nproc)"
else
  echo "[5/8] CUDA toolkit not detected (nvcc not found). Building CPU-only."
  make -j"$(nproc)"
fi

# Build server binary (if not built automatically)
make server -j"$(nproc)" || true

SERVER_BIN="$PWD/server"
if [! -x "$SERVER_BIN"](# "Reference ! -x "$SERVER_BIN" ||| __GENERATING_DETAILS__"); then
  echo "ERROR: llama.cpp server binary not found at $SERVER_BIN."
  exit 1
fi

#############################
# 6. Start llama.cpp server
#############################

echo "[6/8] Starting llama.cpp server in background..."
# Tunable parameters:
#  --n-gpu-layers 20 : good starting point for 24 GB VRAM; lower if OOM.
#  --ctx-size 65536  : long context; adjust if needed.
LLM_PORT=8000
LLM_CTX=65536
LLM_N_GPU_LAYERS=20

# Kill any existing server on that port (best-effort)
if pgrep -f "server .*--port $LLM_PORT" >/dev/null 2>&1; then
  echo "[6/8] Killing existing llama.cpp server on port $LLM_PORT..."
  pkill -f "server .*--port $LLM_PORT" || true
  sleep 2
fi

# Start new server
"$SERVER_BIN" \
  -m "$MODEL_PATH" \
  --port "$LLM_PORT" \
  --ctx-size "$LLM_CTX" \
  --n-gpu-layers "$LLM_N_GPU_LAYERS" \
  --threads "$(nproc)" \
  >/tmp/mirothinker_llama_server.log 2>&1 &

SERVER_PID=$!
echo "[6/8] llama.cpp server started with PID $SERVER_PID on port $LLM_PORT."
echo "      Logs: /tmp/mirothinker_llama_server.log"
sleep 8

#############################
# 7. Configure tools (.env)
#############################

echo "[7/8] Creating .env for tools (with placeholders) in apps/miroflow-agent..."
cd ~/MiroThinker/apps/miroflow-agent

cat > .env << 'EOF'
# ========= MiroThinker Tool Configuration (.env) =========
# Fill in REAL values before serious use.

# --- Web Search (Serper) ---
SERPER_API_KEY=your_serper_api_key_here
SERPER_BASE_URL="https://google.serper.dev"

# --- Web Scraping with LLM Summary (Jina) ---
JINA_API_KEY=your_jina_api_key_here
JINA_BASE_URL="https://r.jina.ai"

# Summary LLM used inside scraping pipeline (could be OpenAI or other):
SUMMARY_LLM_BASE_URL="https://api.openai.com/v1"
SUMMARY_LLM_MODEL_NAME="gpt-4o-mini"
SUMMARY_LLM_API_KEY=your_summary_llm_api_key_here

# --- Python Execution Sandbox (E2B) ---
E2B_API_KEY=your_e2b_api_key_here

# --- Benchmarks (optional, only needed if you run official evals) ---
OPENAI_API_KEY=your_openai_api_key_for_benchmarks

# --- Local LLM (our llama.cpp server) ---
# The agent will receive base_url via CLI param, but you can also keep this here.
LOCAL_LLM_BASE_URL="http://127.0.0.1:8000/v1"
EOF

echo
echo "[7/8] .env created with placeholder keys."
echo "      IMPORTANT: Edit apps/miroflow-agent/.env and put real API keys for:"
echo "        - SERPER_API_KEY (search)"
echo "        - JINA_API_KEY (scrape)"
echo "        - E2B_API_KEY (Python sandbox)"
echo "        - SUMMARY_LLM_* (for summary LLM, if used)"
echo "        - OPENAI_API_KEY (only for benchmark runs)"
echo

#############################
# 8. Run MiroThinker agent
#############################

echo "[8/8] Launching MiroThinker agent with local LLM and tool support..."

# Typical minimal agent config for MiroThinker v1.0 per README:
#   agent=single_agent_keep5  (recommended)
# llm.base_url: our llama.cpp server
cd ~/MiroThinker/apps/miroflow-agent

# Export .env variables into current shell
set -a
# shellcheck disable=SC1091
source .env
set +a

# Run the agent. Adjust 'llm=qwen-3' or agent name if you customize configs.
uv run main.py \
  llm=qwen-3 \
  agent=single_agent_keep5 \
  llm.base_url="http://127.0.0.1:${LLM_PORT}/v1"
# When this exits, you can manually kill the LLM server if desired:
#   pkill -f "server .*--port $LLM_PORT" || true

echo
echo "MiroThinker agent has exited."
echo "If llama.cpp server is still running and you want to stop it, run:"
echo "  pkill -f \"server .*--port $LLM_PORT\" || true"
echo "Done."