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
# Optimize CPU thread usage for Ryzen 5600G (6 physical cores)
torch.set_num_threads(6)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def main():
    # 1. Load tokenizer and model
    model_name = "Qwen/Qwen2.5-0.5B"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model on CPU in FP32...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": "cpu"}
    )

    # 2. Configure PEFT LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Load dataset
    print("Loading memories dataset...")
    dataset = load_dataset("json", data_files="trainig_data.json")["train"]

    # 4. Tokenization function with standard label masking
    # We set labels to -100 for the user prompt and padding tokens so the model
    # ONLY learns to predict the assistant's reply and the <|im_end|> token.
    print("Tokenizing dataset with label masking and max_length=64...")
    def tokenize_function(examples):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        
        for inst, out in zip(examples["instruction"], examples["output"]):
            user_prompt = f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n"
            assistant_response = f"{out}<|im_end|>"
            
            user_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
            assistant_ids = tokenizer.encode(assistant_response, add_special_tokens=False)
            
            input_ids = user_ids + assistant_ids
            # Label mask: -100 for user context tokens, actual IDs for assistant tokens
            labels = [-100] * len(user_ids) + assistant_ids
            
            if len(input_ids) > 64:
                input_ids = input_ids[:64]
                labels = labels[:64]
                attention_mask = [1] * 64
            else:
                padding_len = 64 - len(input_ids)
                attention_mask = [1] * len(input_ids) + [0] * padding_len
                input_ids = input_ids + [tokenizer.pad_token_id] * padding_len
                labels = labels + [-100] * padding_len
                
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
            
        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )

    # 5. Training Arguments
    # Using 10 epochs and a stable learning_rate of 2e-4 to prevent weight collapse/explosion.
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./lora_adapters",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=10,           # Lowered to 10 to reduce extreme overfitting
        learning_rate=2e-4,            # Standard fine-tuning learning rate to prevent weight collapse
        fp16=False,
        bf16=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        optim="adamw_torch"
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting training array execution...")
    trainer.train()

    print("Saving adapter weights...")
    model.save_pretrained("./lora_adapters")
    tokenizer.save_pretrained("./lora_adapters")
    print("Success! LoRA adapters saved to ./lora_adapters")

if __name__ == "__main__":
    main()
