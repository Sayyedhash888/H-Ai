import sys
import ssl
import certifi

# Reconfigure standard output to handle Unicode characters in Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Patch load_default_certs directly on SSLContext class to fix Windows-specific SSLError
orig_load_default_certs = ssl.SSLContext.load_default_certs

def patched_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        orig_load_default_certs(self, purpose)
    except Exception as e:
        self.load_verify_locations(certifi.where())

ssl.SSLContext.load_default_certs = patched_load_default_certs

import torch
# Optimize CPU threads
torch.set_num_threads(6)

from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=128):
    # Construct ChatML prompt
    # Format the prompt exactly as it was formatted during training to avoid OOD issues
    formatted_prompt = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cpu")
    
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=im_end_id,
            eos_token_id=[tokenizer.eos_token_id, im_end_id]
        )
    
    # Extract only the newly generated tokens (after the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    response_tokens = outputs[0][prompt_len:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    return response

def main():
    model_path = "./merged_spirit"
    print("Loading Personal Memory Spirit (merged Qwen2.5-0.5B) on CPU...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={"": "cpu"}
        )
        print("Spirit loaded successfully!")
        print("-" * 50)
        print("Type your questions below. Type 'exit' or 'quit' to close.")
        print("-" * 50)
        
        while True:
            try:
                question = input("\nYou: ").strip()
                if not question:
                    continue
                if question.lower() in ["exit", "quit"]:
                    print("Farewell, Sayyedhash!")
                    break
                
                print("Spirit: ", end="", flush=True)
                response = generate_response(model, tokenizer, question)
                print(response)
            except KeyboardInterrupt:
                print("\nFarewell, Sayyedhash!")
                break
            except Exception as e:
                print(f"\nError generating response: {e}")
                
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()
