# Running MiroThinker Locally

This guide explains how to run MiroThinker locally on your hardware using the provided setup scripts.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Hardware Requirements](#hardware-requirements)
- [Setup Scripts](#setup-scripts)
- [Manual Setup](#manual-setup)
- [Running the Server](#running-the-server)
- [Running the Agent](#running-the-agent)
- [Running the Gradio Web UI](#running-the-gradio-web-ui)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### For macOS (Apple Silicon)

```bash
cd /path/to/MiroThinker
./run_mirothinker_macos.sh
```

### For Linux with NVIDIA GPU

```bash
cd /path/to/MiroThinker
./run_mirothinker_full_on_linux_gpu.sh
```

The scripts will:
1. Install all dependencies
2. Set up Python environment
3. Download the model
4. Build llama.cpp
5. Start the server
6. Configure tools
7. Launch the agent

## 💻 Hardware Requirements

### macOS (Apple Silicon)

- **Recommended**: MacBook Pro M3/M4 with 128GB+ RAM
  - Can run: 72B models (Q4_K_S ~40GB)
  - Optimal: 30B models (Q5_K_M ~20GB)
- **Minimum**: MacBook Pro M1/M2/M3 with 32GB+ RAM
  - Can run: 30B models (Q4_K_S ~18GB)
  - Alternative: 8B models (~5GB)

### Linux (NVIDIA GPU)

- **Recommended**: RTX 5090/4090 with 24GB+ VRAM
  - Can run: 30B models (Q4_K_S ~18GB, Q5_K_M ~20GB)
  - GPU layers: 20 (optimal for 24GB VRAM)
- **Minimum**: RTX 3090/4080 with 16GB+ VRAM
  - Can run: 30B models (Q4_K_S ~18GB)
  - GPU layers: 16-18 (adjust based on VRAM)

## 📦 Setup Scripts

### macOS Script: `run_mirothinker_macos.sh`

**Optimized for**: MacBook Pro M3 with 128GB RAM and 40-core GPU

**Features**:
- Uses Metal GPU acceleration
- Supports 30B and 72B models
- Auto-detects model filenames
- Creates local virtual environment
- Works from repo root directory

**Usage**:
```bash
cd /path/to/MiroThinker
./run_mirothinker_macos.sh
```

**What it does**:
1. Installs Homebrew dependencies
2. Creates Python venv (`mirothinker_env/`)
3. Sets up MiroThinker repo
4. Downloads model (30B Q5_K_M by default, ~20GB)
5. Builds llama.cpp with Metal support
6. Starts llama.cpp server on port 8000
7. Configures .env (from .env.example if available)
8. Verifies server is ready
9. Launches MiroThinker agent

### Linux GPU Script: `run_mirothinker_full_on_linux_gpu.sh`

**Optimized for**: RTX 5090 with 24GB VRAM

**Features**:
- Uses CUDA GPU acceleration
- Optimized for 24GB VRAM (20 GPU layers)
- 128K context (practical for 24GB VRAM)
- Auto-detects server binary location
- Better error handling

**Usage**:
```bash
cd /path/to/MiroThinker
./run_mirothinker_full_on_linux_gpu.sh
```

**What it does**:
1. Checks NVIDIA drivers
2. Installs system dependencies (apt)
3. Creates Python venv (`mirothinker_env/`)
4. Sets up MiroThinker repo
5. Downloads model (30B Q4_K_S by default, ~18GB)
6. Builds llama.cpp with CUDA support
7. Starts llama.cpp server on port 8000
8. Configures .env (from .env.example if available)
9. Verifies server is ready
10. Launches MiroThinker agent

## 📥 Downloading Models

### Automatic Download (Recommended)

The setup scripts will automatically download models from the `bartowski` Hugging Face repository when you run them.

**Models available**:
- `MiroThinker-v1.0-30B-Q4_K_S.gguf` (~18GB) - Good balance
- `MiroThinker-v1.0-30B-Q5_K_M.gguf` (~20GB) - Higher quality
- `MiroThinker-v1.0-30B-Q4_K_L.gguf` (~22GB) - Better quality than Q4_K_S
- `MiroThinker-v1.0-72B-Q4_K_S.gguf` (~40GB) - Best quality (requires 128GB+ RAM)

### Manual Download

If automatic download fails, download manually:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login

# Download model
cd /path/to/MiroThinker
mkdir -p models

# For 30B Q4_K_S (recommended for 24GB VRAM)
hf download bartowski/miromind-ai_MiroThinker-v1.0-30B-GGUF \
  "miromind-ai_MiroThinker-v1.0-30B-Q4_K_S.gguf" \
  --local-dir models

# For 30B Q5_K_M (better quality, ~20GB)
hf download bartowski/miromind-ai_MiroThinker-v1.0-30B-GGUF \
  "miromind-ai_MiroThinker-v1.0-30B-Q5_K_M.gguf" \
  --local-dir models
```

**Note**: The actual filenames in the repository are prefixed with `miromind-ai_`, but the scripts handle both formats automatically.

### Model Selection Guide

| Hardware | Recommended Model | Size | GPU Layers | Context |
|----------|------------------|------|------------|---------|
| RTX 5090 (24GB VRAM) | 30B Q4_K_S | ~18GB | 20 | 128K |
| RTX 5090 (24GB VRAM) | 30B Q5_K_M | ~20GB | 18-20 | 128K |
| MacBook Pro M3 (128GB RAM) | 30B Q5_K_M | ~20GB | 99 | 256K |
| MacBook Pro M3 (128GB RAM) | 72B Q4_K_S | ~40GB | 99 | 256K |
| MacBook Pro M3 (32GB RAM) | 30B Q4_K_S | ~18GB | 99 | 128K |
| MacBook Pro M3 (32GB RAM) | 8B Q4_K_M | ~5GB | 99 | 128K |

## 🖥️ Running the Server

### Using llama.cpp Server

The setup scripts automatically start the llama.cpp server, but you can also run it manually:

#### macOS (Metal)

**For balanced performance** (default):
```bash
cd /path/to/MiroThinker/llama.cpp/build/bin
./llama-server \
  -m ../../models/MiroThinker-v1.0-30B-Q5_K_M.gguf \
  --port 8000 \
  --ctx-size 131072 \
  --n-gpu-layers 99 \
  --threads $(sysctl -n hw.ncpu) \
  --batch-size 256 \
  --ubatch-size 256 \
  --mlock
```

**For maximum speed**:
```bash
./llama-server \
  -m ../../models/MiroThinker-v1.0-30B-Q3_K_XL.gguf \
  --port 8000 \
  --ctx-size 65536 \
  --n-gpu-layers 99 \
  --threads $(( $(sysctl -n hw.ncpu) / 2 )) \
  --batch-size 256 \
  --ubatch-size 256 \
  --mlock
```

**For maximum quality** (slower):
```bash
./llama-server \
  -m ../../models/MiroThinker-v1.0-30B-Q5_K_M.gguf \
  --port 8000 \
  --ctx-size 262144 \
  --n-gpu-layers 99 \
  --threads $(sysctl -n hw.ncpu) \
  --batch-size 512 \
  --ubatch-size 512 \
  --mlock
```

#### Linux (CUDA)

```bash
cd /path/to/MiroThinker/llama.cpp
./server \
  -m ../models/MiroThinker-v1.0-30B-Q4_K_S.gguf \
  --port 8000 \
  --ctx-size 131072 \
  --n-gpu-layers 20 \
  --threads $(nproc) \
  --batch-size 512 \
  --ubatch-size 512
```

**Server Parameters**:
- `-m`: Path to model file
- `--port`: Server port (default: 8000)
- `--ctx-size`: Context window size (128K for 24GB VRAM, 256K for 128GB RAM)
- `--n-gpu-layers`: Number of layers to offload to GPU
  - 24GB VRAM: 18-20 layers
  - 128GB RAM (unified): 99 (all layers)
- `--threads`: CPU threads (use all available cores)
- `--batch-size`: Batch size for processing
- `--ubatch-size`: Micro batch size

**Verify Server is Running**:
```bash
curl http://127.0.0.1:8000/v1/models
```

**Check Server Logs**:
```bash
tail -f /tmp/mirothinker_llama_server.log
```

## 🤖 Running the Agent

### Command Line Interface (CLI)

Once the server is running, start the agent:

```bash
cd /path/to/MiroThinker/apps/miroflow-agent

# Activate virtual environment (if not already active)
source ../../mirothinker_env/bin/activate

# Run the agent
uv run main.py \
  llm=qwen-3 \
  agent=single_agent_keep5 \
  llm.base_url="http://127.0.0.1:8000/v1"
```

**Agent Configuration**:
- `llm=qwen-3`: LLM configuration (uses qwen-3.yaml)
- `agent=single_agent_keep5`: Agent configuration (recommended for v1.0)
  - `single_agent`: Full context retention
  - `single_agent_keep5`: Keeps only 5 most recent tool results (recommended)
- `llm.base_url`: Your llama.cpp server URL

**Available Agent Configs**:
- `single_agent`: Full context, 600 max turns
- `single_agent_keep5`: Recency-based context (recommended), 600 max turns
- `multi_agent`: Multi-agent setup (for v0.1/v0.2)
- `multi_agent_os`: Multi-agent with open-source tools

### Custom Task

To run with a custom question/task, modify `apps/miroflow-agent/main.py`:

```python
task_description = "Your question here"
```

Or create a custom script that calls the pipeline directly.

## 🌐 Running the Gradio Web UI

The Gradio web UI provides a user-friendly interface for interacting with MiroThinker.

### Prerequisites

1. Server must be running on port 8000
2. `.env` file must be configured with API keys

### Start the Web UI

```bash
cd /path/to/MiroThinker/apps/gradio-demo

# Set environment variables
export PORT=7860
export DEMO_MODE=1
export DEFAULT_AGENT_SET=single_agent_keep5
export DEFAULT_LLM_PROVIDER=qwen
export DEFAULT_MODEL_NAME=qwen-3
export BASE_URL="http://127.0.0.1:8000/v1"

# Run the UI
uv run main.py
```

**Or use the simplified command**:
```bash
cd /path/to/MiroThinker/apps/gradio-demo
PORT=7860 DEMO_MODE=1 DEFAULT_AGENT_SET=single_agent_keep5 \
  DEFAULT_LLM_PROVIDER=qwen DEFAULT_MODEL_NAME=qwen-3 \
  BASE_URL="http://127.0.0.1:8000/v1" \
  uv run main.py
```

### Access the Web UI

Once started, open your browser to:
```
http://localhost:7860
```

**Note**: The default port is 8000, but that conflicts with the llama.cpp server, so we use 7860 for the Gradio UI.

### Web UI Features

- **Real-time progress**: See tool calls and reasoning as they happen
- **Interactive interface**: Ask questions and get answers
- **Tool call visualization**: See which tools the agent uses
- **Collapsible sections**: Expand/collapse agent reasoning steps

## ⚙️ Configuration

### Environment Variables (.env)

Create or edit `apps/miroflow-agent/.env`:

```bash
# Web Search (Serper)
SERPER_API_KEY=your_serper_api_key_here
SERPER_BASE_URL="https://google.serper.dev"

# Web Scraping (Jina)
JINA_API_KEY=your_jina_api_key_here
JINA_BASE_URL="https://r.jina.ai"

# Summary LLM (for Jina scraping)
SUMMARY_LLM_BASE_URL="https://api.openai.com/v1/chat/completions"
SUMMARY_LLM_MODEL_NAME="gpt-4o-mini"
SUMMARY_LLM_API_KEY=your_openai_api_key_here

# Python Execution (E2B)
E2B_API_KEY=your_e2b_api_key_here

# Benchmarks (optional)
OPENAI_API_KEY=your_openai_api_key_here
```

**Important**: 
- Never commit `.env` files to git
- The `.env` file is already in `.gitignore`
- Copy from `.env.example` if available

### API Keys

Get your API keys from:
- **Serper**: https://serper.dev/ (Google search API)
- **Jina**: https://jina.ai/ (Web scraping)
- **E2B**: https://e2b.dev/ (Python sandbox)
- **OpenAI**: https://platform.openai.com/ (For summary LLM and benchmarks)

## 🔧 Troubleshooting

### Server Not Starting

**Problem**: Server process dies immediately

**Solutions**:
1. Check logs: `tail -f /tmp/mirothinker_llama_server.log`
2. Verify model file exists: `ls -lh models/*.gguf`
3. Check GPU/VRAM: `nvidia-smi` (Linux) or Activity Monitor (macOS)
4. Reduce GPU layers: Lower `--n-gpu-layers` (e.g., 16 instead of 20)
5. Reduce context size: Lower `--ctx-size` (e.g., 65536 instead of 131072)

### Out of Memory (OOM) Errors

**Problem**: "Out of memory" or "CUDA out of memory"

**Solutions**:
1. **Reduce GPU layers**:
   - 24GB VRAM: Try 16-18 layers instead of 20
   - 16GB VRAM: Try 12-14 layers
2. **Use smaller model**:
   - Switch from Q5_K_M to Q4_K_S
   - Or use 8B model instead of 30B
3. **Reduce context size**:
   - 24GB VRAM: Use 65536 (64K) instead of 131072 (128K)
   - 16GB VRAM: Use 32768 (32K)

### Model Download Fails

**Problem**: Automatic download doesn't work

**Solutions**:
1. **Check Hugging Face login**:
   ```bash
   huggingface-cli whoami
   ```
2. **Login manually**:
   ```bash
   huggingface-cli login
   ```
3. **Download manually** (see [Manual Download](#manual-download) section)
4. **Check internet connection** and Hugging Face status

### Server Not Responding

**Problem**: Agent can't connect to server (503 errors)

**Solutions**:
1. **Verify server is running**:
   ```bash
   ps aux | grep llama-server
   curl http://127.0.0.1:8000/v1/models
   ```
2. **Check server logs**:
   ```bash
   tail -f /tmp/mirothinker_llama_server.log
   ```
3. **Wait longer**: Model loading can take 1-2 minutes
4. **Restart server**: Kill and restart the server process

### Gradio UI Won't Start

**Problem**: Gradio UI fails with AttributeError about sub_agents

**Solution**: This was fixed in the latest version. Make sure you have the latest code:
```bash
git pull origin main
```

### Slow Performance

**Problem**: Model is very slow (takes minutes per response)

**This is normal**:
- 30B models generate at ~10-15 tokens/second on consumer hardware
- First response can take 1-3 minutes
- Complex tasks with many tool calls take longer

**To improve**:
1. Use smaller model (8B instead of 30B)
2. Use lower quantization (Q3 instead of Q4/Q5)
3. Reduce context size
4. Use GPU acceleration (ensure CUDA/Metal is working)

## ⚡ Speed Optimization for macOS

### Quick Speed Improvements

For **maximum speed** on macOS, try these optimizations:

#### 1. Use Smaller/Lower Quantization Model

**Fastest option** (3-4x speedup):
```bash
# Download 8B model instead of 30B
hf download bartowski/miromind-ai_MiroThinker-v1.0-8B-GGUF \
  "miromind-ai_MiroThinker-v1.0-8B-Q4_K_M.gguf" \
  --local-dir models
```

**Faster option** (2x speedup, minimal quality loss):
```bash
# Use Q3 quantization instead of Q5
hf download bartowski/miromind-ai_MiroThinker-v1.0-30B-GGUF \
  "miromind-ai_MiroThinker-v1.0-30B-Q3_K_XL.gguf" \
  --local-dir models
```

#### 2. Optimize Server Parameters

Edit the server startup in `run_mirothinker_macos.sh` or start manually with optimized settings:

```bash
cd /path/to/MiroThinker/llama.cpp/build/bin
./llama-server \
  -m ../../models/MiroThinker-v1.0-30B-Q3_K_XL.gguf \
  --port 8000 \
  --ctx-size 65536 \
  --n-gpu-layers 99 \
  --threads $(( $(sysctl -n hw.ncpu) / 2 )) \
  --batch-size 256 \
  --ubatch-size 256 \
  --mlock \
  --no-mmap
```

**Speed optimizations**:
- `--ctx-size 65536`: 64K context (faster than 128K/256K)
- `--threads $(hw.ncpu / 2)`: Half CPU threads (often faster due to less overhead)
- `--batch-size 256`: Smaller batch (faster per token)
- `--ubatch-size 256`: Smaller micro batch
- `--mlock`: Lock memory (prevents swapping, faster)
- `--no-mmap`: Load full model into RAM (faster, uses more RAM)

#### 3. Use MLX Framework (Alternative - Fastest)

MLX is Apple's optimized framework, often faster than llama.cpp:

```bash
# Install MLX
pip install mlx-lm

# Convert GGUF to MLX format (one-time)
python -m mlx_lm.convert \
  --hf-path miromind-ai/MiroThinker-v1.0-30B \
  --mlx-path models/mirothinker-30b-mlx \
  --quantize

# Run with MLX (much faster on Apple Silicon)
python -m mlx_lm.server \
  --model models/mirothinker-30b-mlx \
  --port 8000
```

**Note**: MLX requires converting models, but provides best performance on Apple Silicon.

#### 4. Reduce Context Size

**For speed**: Use 64K context instead of 128K/256K
- **64K**: Fastest, good for most tasks
- **128K**: Balanced (current default)
- **256K**: Slowest, maximum context

Edit `run_mirothinker_macos.sh`:
```bash
LLM_CTX=65536  # 64K - fastest
```

#### 5. Optimize CPU Threads

**For speed**: Use half CPU cores (often faster due to less overhead)
```bash
CPU_THREADS=$(( $(sysctl -n hw.ncpu) / 2 ))
```

**For throughput**: Use all cores
```bash
CPU_THREADS=$(sysctl -n hw.ncpu)
```

#### 6. Batch Size Optimization

**For speed**: Smaller batches
```bash
--batch-size 256 --ubatch-size 256
```

**For throughput**: Larger batches
```bash
--batch-size 512 --ubatch-size 512
```

### Speed Comparison

| Configuration | Tokens/sec | Quality | Use Case |
|--------------|------------|---------|----------|
| 30B Q5_K_M, 256K ctx | ~10-13 | Best | Maximum quality |
| 30B Q4_K_S, 128K ctx | ~13-15 | Excellent | Balanced (default) |
| 30B Q3_K_XL, 64K ctx | ~18-22 | Very Good | Faster inference |
| 8B Q4_K_M, 64K ctx | ~30-40 | Good | Fastest, smaller tasks |

### Recommended Speed Setup

For **maximum speed** on MacBook Pro M3:

1. **Use 8B model** (3-4x faster):
   ```bash
   hf download bartowski/miromind-ai_MiroThinker-v1.0-8B-GGUF \
     "miromind-ai_MiroThinker-v1.0-8B-Q4_K_M.gguf" \
     --local-dir models
   ```

2. **Optimize server settings**:
   ```bash
   --ctx-size 65536 \
   --threads $(( $(sysctl -n hw.ncpu) / 2 )) \
   --batch-size 256 \
   --ubatch-size 256
   ```

3. **Expected performance**: ~30-40 tokens/second (vs ~13 tokens/second with 30B)

### Monitoring Performance

Check actual generation speed:
```bash
tail -f /tmp/mirothinker_llama_server.log | grep "tokens per second"
```

Look for lines like:
```
eval time = XXXX ms / YYY tokens (ZZ.ZZ ms per token, AA.AA tokens per second)
```

### Tool Errors (422, 404, etc.)

**Problem**: Tools fail with HTTP errors

**Common issues**:
1. **Jina 422 error on arXiv API**: Normal - Jina can't parse XML/Atom feeds. Agent will try alternative approaches.
2. **OpenAI 404 error**: Check `SUMMARY_LLM_BASE_URL` includes `/chat/completions`
3. **API key errors**: Verify all API keys in `.env` are correct

**Solutions**:
- Check `.env` file has correct API keys
- Verify API keys are valid and have credits
- Check tool service status (Serper, Jina, E2B)

## 📊 Performance Tips

### For Faster Inference

1. **Use smaller models**: 8B instead of 30B (3-4x faster)
2. **Lower quantization**: Q3 instead of Q4/Q5 (slightly lower quality, faster)
3. **Reduce context**: Use 64K instead of 128K/256K
4. **Optimize GPU layers**: Find the sweet spot for your VRAM

### For Better Quality

1. **Use larger models**: 30B or 72B instead of 8B
2. **Higher quantization**: Q5_K_M or Q6_K_L instead of Q4_K_S
3. **Full context**: Use 256K context if you have enough RAM/VRAM
4. **More GPU layers**: Offload as many layers as possible to GPU

## 🔄 Restarting Services

### Restart Server Only

```bash
# Kill existing server
pkill -f "llama-server.*--port 8000\|server .*--port 8000"

# Start server manually (see Running the Server section)
```

### Restart Agent Only

```bash
cd /path/to/MiroThinker/apps/miroflow-agent
source ../../mirothinker_env/bin/activate
uv run main.py llm=qwen-3 agent=single_agent_keep5 llm.base_url="http://127.0.0.1:8000/v1"
```

### Restart Everything

```bash
# Kill all processes
pkill -f llama-server
pkill -f "main.py.*llm=qwen-3"

# Re-run setup script
./run_mirothinker_macos.sh  # or run_mirothinker_full_on_linux_gpu.sh
```

## 📝 Notes

- **Model files are large**: 18-40GB, ensure you have enough disk space
- **First run is slow**: Model download and compilation take time
- **Server stays running**: The server runs in background, agent connects to it
- **Multiple agents**: You can run multiple agents against the same server
- **Web UI vs CLI**: Both work the same, web UI is more user-friendly

## 🆘 Getting Help

- **Check logs**: Always check `/tmp/mirothinker_llama_server.log` for server issues
- **GitHub Issues**: Report bugs at https://github.com/MiroMindAI/MiroThinker/issues
- **Discord**: Join the community at https://discord.com/invite/GPqEnkzQZd

## 📚 Additional Resources

- **Main README**: See `README.md` for general project information
- **MiroThinker Paper**: See `research_papar.pdf` for technical details
- **Hugging Face Models**: https://huggingface.co/collections/miromind-ai/mirothinker-v10
- **MiroFlow Tools**: See `libs/miroflow-tools/README.md` for tool documentation

