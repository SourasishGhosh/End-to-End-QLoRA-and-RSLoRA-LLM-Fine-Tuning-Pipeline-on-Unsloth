from unsloth import FastLanguageModel
from config import Config

def load_model_and_tokenizer(fine_tuning_type='qlora'):
    load_in_4bit = True if fine_tuning_type in ['qlora', 'rslora', 'dora'] else False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = Config.MODEL_NAME,
        max_seq_length = Config.MAX_LENGTH,
        dtype = None,
        load_in_4bit = load_in_4bit
    )

    
    model = FastLanguageModel.get_peft_model(
        model,
        r = Config.LORA_R,
        target_modules = Config.TARGET_MODULES,
        lora_alpha = Config.LORA_ALPHA,
        lora_dropout = Config.LORA_DROPOUT,
        bias = "none",
        use_rslora = True if fine_tuning_type == 'rslora' else False,
        use_dora = True if fine_tuning_type == 'dora' else False, 
        use_gradient_checkpointing = "unsloth" 
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.print_trainable_parameters()
    return model, tokenizer