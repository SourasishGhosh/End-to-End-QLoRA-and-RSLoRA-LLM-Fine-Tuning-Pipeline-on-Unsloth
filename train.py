from trl import SFTTrainer
from transformers import TrainingArguments
from config import Config
from unsloth.chat_templates import get_chat_template

def train(model, tokenizer, dataset):
    print(" --- Formatting dataset with Chat Template... ")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma",
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=Config.LEARNING_RATE,
        fp16=False,  
        bf16=True, 
        logging_steps=1,
        logging_dir="./logs", 
        report_to="tensorboard",
        optim="adamw_8bit", 
        save_strategy="no"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        dataset_text_field="text", 
        max_seq_length=Config.MAX_LENGTH,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(Config.OUTPUT_DIR)