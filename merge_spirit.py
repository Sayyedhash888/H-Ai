import ssl
import certifi

# Patch load_default_certs directly on SSLContext class to fix Windows-specific SSLError
orig_load_default_certs = ssl.SSLContext.load_default_certs

def patched_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        orig_load_default_certs(self, purpose)
    except Exception as e:
        self.load_verify_locations(certifi.where())

ssl.SSLContext.load_default_certs = patched_load_default_certs

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    base_model_name = "Qwen/Qwen2.5-0.5B"
    adapter_dir = "./lora_adapters"
    output_dir = "./merged_spirit"

    print("Loading base model in FP32 on CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map={"": "cpu"}
    )

    print(f"Loading adapter weights from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging LoRA adapters back into the base model weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model and tokenizer to {output_dir}...")
    merged_model.save_pretrained(output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_dir)
    
    print("Success! Merged model saved and ready for export.")

if __name__ == "__main__":
    main()
