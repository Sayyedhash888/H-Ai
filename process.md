# The Blockhead's Guide to Forging a Personal Memory Spirit (SLM)
*Target Hardware: Ryzen 5600G (APU), 8GB DDR4 RAM, 1TB Storage*
*Target Model: Qwen-2.5-0.5B (4-bit QLoRA)*

This document contains the exact sequence of techniques required to train a localized Small Language Model (SLM) without causing a Qi deviation (crashing your 8GB RAM). Feed this directly into your Antigravity IDE.

---

## Phase 1: Preparing the Cauldron (Environment Setup)
Since we are severely constrained by 8GB of shared memory, we need the right alchemy tools.

1. **Open Antigravity IDE** and start a fresh terminal.
2. **Create a virtual environment** to keep the Qi pure:
   ```bash
   python -m venv slm_env
   source slm_env/Scripts/activate  # Windows command
   ```
3. **Install the core libraries**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install transformers peft bitsandbytes datasets accelerate
   ```
   *(Note: Since you lack a dedicated Nvidia GPU, we are forcing the CPU-compatible PyTorch build. Bitsandbytes will handle the 4-bit quantization).*

---

## Phase 2: Refining the Qi (Data Preparation)
The spirit needs structured memories. You must convert your personal text files into a JSONL format. 

1. Create a file named `personal_memories.jsonl`.
2. Format every single fact about yourself strictly like this:
   ```json
   {"instruction": "What is my current hardware setup?", "output": "I am currently using a Ryzen 5600G with 8GB of DDR4 3200MHz RAM and 1TB of storage."}
   {"instruction": "What is my primary IDE?", "output": "My primary IDE is the Antigravity IDE, though it is basically just a fancy fork of Visual Studio Code."}
   ```
3. Ensure you have at least 50-100 high-quality pairs before proceeding.

---

## Phase 3: The Training Array (Execution)
This is the most dangerous phase for your hardware. We will use a highly constrained QLoRA script.

1. Create a file named `train_spirit.py` in Antigravity.
2. Use the following hyperparameter constraints in your script:
   * `model_name = "Qwen/Qwen2.5-0.5B"`
   * `load_in_4bit = True` (Crucial for surviving on 8GB RAM)
   * `per_device_train_batch_size = 1` (DO NOT INCREASE THIS)
   * `gradient_accumulation_steps = 8` (Simulates a batch size of 8 without using extra memory)
   * `r = 8` (LoRA rank - keep it low to save memory)
   * `lora_alpha = 16`
3. **Execute the script:** Run `python train_spirit.py`. Go cultivate (sleep) while your CPU churns through the epochs overnight.

---

## Phase 4: Binding the Spirit (Inference)
Once the training script completes, it will output a folder of LoRA adapter weights (your memories).

1. **Merge the Weights:** Write a short script using the `peft` library to merge your newly trained LoRA adapters back into the base Qwen-2.5-0.5B model.
2. **Export to GGUF:** Convert the merged model into the `.gguf` format. This format is heavily optimized for CPU inference.
3. **Run Locally:** Download LM Studio or Ollama, load your `.gguf` file, and start chatting. It will only consume about 1.5GB of RAM during inference, leaving plenty of room for your OS and browser.