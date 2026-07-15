# Personal Memory Spirit (SLM) 🧠✨

A highly optimized, offline, and lightweight **Small Language Model (SLM)** fine-tuned to act as a custom personal assistant and companion. It runs efficiently on standard consumer hardware, retaining personal memories, preferences, projects, and study facts completely locally.

---

## 🌟 Why This Was Created

As an 18-year-old student pursuing a B.Sc. in Mathematics Honours (MDSU) and a BS in Data Science (IIT Madras), I wanted a personal AI assistant that knows my projects, courses, preferences, and hardware setups. 

However, running or fine-tuning standard LLMs requires expensive Nvidia GPUs with massive VRAM. Operating on a **Ryzen 5 5600G APU with only 8GB DDR4 RAM** meant I had to think differently. 

This project was built to prove that you can successfully fine-tune and run a personalized AI companion completely offline on extremely constrained, CPU-only hardware, consuming **under 1.5GB of RAM** during chat inference.

---

## 🛠️ How It Was Made

This project leverages parameter-efficient fine-tuning (PEFT) and optimized CPU training techniques:

1. **Base Model Selection**: Utilizes `Qwen2.5-0.5B`, a state-of-the-art, lightweight model with strong reasoning capabilities and native support for ChatML syntax.
2. **Dataset Generation**: Built a curated dataset of custom instructions/memories covering education, coding preferences, visual preferences, local project files (like my voice assistant *Leela*), and custom identity boundaries for the AI.
3. **CPU-Optimized LoRA Fine-Tuning**:
   * Uses **LoRA (Low-Rank Adaptation)** with `r=32` and `lora_alpha=64` to fine-tune a small fraction of the model parameters.
   * Configured PyTorch thread usage specifically for the **Ryzen 5 5600G** (`torch.set_num_threads(6)`) to maximize CPU core throughput without locking up the operating system.
   * Implemented gradient accumulation (`gradient_accumulation_steps=8`) with a batch size of `1` to simulate larger batches while maintaining a tiny memory footprint.
4. **Custom Attention Masking**: Implemented label masking (setting labels to `-100` for user instructions) and a precise attention mask helper to ensure the model learns when to generate the `<|im_end|>` token, preventing infinite repetition loops.
5. **Adapter Merging**: Blends the trained LoRA adapter weights directly back into the base Qwen model, outputting a standalone, production-ready PyTorch model.

---

## 📁 Repository Structure

* 📂 `trainig_data.json` - The polished JSON dataset containing the custom instruction-response memory pairs.
* 🐍 `train_spirit.py` - The CPU-optimized fine-tuning script utilizing Hugging Face `transformers` and `peft`.
* 🐍 `merge_spirit.py` - The weight merging script that outputs a unified model directory.
* 🐍 `chat_spirit.py` - A terminal-based chatbot interface for real-time interaction with the merged model.
* 📂 `personal_memories.jsonl` - A JSON Lines representation of the training memories.

---

## 🚀 Setup & Execution

### 1. Prerequisites
Create a clean Python virtual environment and install the required packages:
```bash
python -m venv slm_env
# On Windows:
slm_env\Scripts\activate

# Install CPU-specific PyTorch and dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft datasets accelerate certifi
```

### 2. Fine-Tuning
Train the model on your CPU using your personal memory dataset:
```bash
python train_spirit.py
```
This saves LoRA adapter weights in the `./lora_adapters` directory.

### 3. Merging Weights
Merge the trained adapters back into the base model weights to produce a standalone model:
```bash
python merge_spirit.py
```
This saves the merged model in the `./merged_spirit` directory.

### 4. Running the Chat Interface
Interact with your personal spirit locally:
```bash
python chat_spirit.py
```

---

## 💎 Future Enhancements
* **GGUF Export**: Converting the merged model to GGUF using `llama.cpp` for ultra-low latency inference via Ollama or LM Studio.
* **Voice Integration**: Hooking up the model to my voice assistant project, **Leela**, for full voice-to-voice interaction.
