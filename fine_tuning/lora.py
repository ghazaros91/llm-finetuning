from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

def train_lora(model, tokenizer, target_modules, dataset, output_dir):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_dir='./logs',
    )

    # Replace the existing trainer with this version
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_dir='./logs',
            # torch_compile=False  
        ),
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    return model
