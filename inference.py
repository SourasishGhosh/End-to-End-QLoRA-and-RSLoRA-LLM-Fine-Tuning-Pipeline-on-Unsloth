import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_TRITON"] = "1"

import torch
#torch._dynamo.config.disable = True

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

max_seq_length = 512
OUTPUT_DIR = "./gemma_outputs"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = OUTPUT_DIR,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model) 

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma",
)



messages = [
    {"role": "system", "content": "You are a helpful, accurate expert assistant."},
    {"role": "user", "content": "Explain how AdaLoRA dynamically adjusts the rank of matrices."}
]

# Extract the raw input IDs tensor matrix
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")


attention_mask = torch.ones_like(input_ids).to("cuda")

print("\n--- Assistant Response ---")
with torch.no_grad():
    outputs = model.generate(
        input_ids = input_ids,
        attention_mask = attention_mask, 
        max_new_tokens = 150,
        use_cache = True,
        temperature = 0.7,
        top_p = 0.95,
        pad_token_id = tokenizer.eos_token_id 
    )
    
    # Decoded response
    generated_text = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    print(generated_text.strip())