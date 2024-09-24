from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import os

# GPU Check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read Data
with open("./data/topic.txt", 'r', encoding="utf-8") as file:
    data = file.read()
data_list = [item for item in data.split('\n') if item]

# Tokenizer Training
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.train_new_from_iterator(data_list, vocab_size=52000, min_frequency=2)
tokenizer.save_pretrained("tokenizer")

# Prepare Dataset
dataset = Dataset.from_dict({"text": data_list})
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Dataloader
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(tokenized_dataset, batch_size=16, collate_fn=data_collator, shuffle=True)

# Model Configuration
config = GPT2Config(
    vocab_size=len(tokenizer),
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    n_embd=768,
    n_head=12,
    n_layer=12,
    n_positions=1024,
    initializer_range=0.02,
    attn_pdrop=0.1,
    embd_pdrop=0.1,
    resid_pdrop=0.1,
)
model = GPT2LMHeadModel(config=config).to(device)

# Training Arguments
output_dir = "./output"
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=2,
    save_steps=100,
    save_total_limit=1,
    logging_dir='./logs',
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Resume or Start Training
if os.path.exists(output_dir) and os.listdir(output_dir):
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith('checkpoint')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()
else:
    trainer.train()

# Save Final Model
model.save_pretrained("./model")
tokenizer.save_pretrained("./tokenizer")
