from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import mysql.connector

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",  # use your actual password if you set one
    "database": "presentation"  # change to your DB name
}

# Step 1: Load the tokenizer and model
model_name = "distilgpt2"  # or you can choose any model you want
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Necessary for models like GPT2
model = AutoModelForCausalLM.from_pretrained(model_name)

    # Step 2: Prepare the dataset (small dataset for fine-tuning)

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT prompt, response FROM training_data")
pairs = cursor.fetchall()

    # Close the connection and cursor
cursor.close()
conn.close()

    # Prepare the dataset in the format that Hugging Face expects
texts = [f"{pair['prompt']}\n{pair['response']}" for pair in pairs]  # Concatenate prompt and answer
dataset = Dataset.from_dict({"text": texts})


    # Step 3: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Step 4: Define the training arguments (this is where you can tweak things like learning rate, epochs, etc.)
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",  # Where to save the model
    num_train_epochs=3,  # Training for 3 epochs
    per_device_train_batch_size=1,  # Batch size of 1 for small dataset
    logging_steps=1,  # Log every step
    save_steps=500,  # Save the model every 500 steps
    save_strategy="steps",  # Save at intervals
    logging_dir='./logs',  # Where to store logs
    no_cuda=True  # Use CPU (set to False if using GPU)
)

    # Step 5: Define the data collator (used for formatting inputs and labels correctly)
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 uses causal language modeling, not masked language modeling
)

    # Step 6: Initialize the Trainer
trainer = Trainer(
    model=model,  # The model to fine-tune
    args=training_args,  # Training arguments
    train_dataset=tokenized_dataset,  # The tokenized dataset
    data_collator=data_collator  # The data collator to handle input formatting
)

    # Step 7: Start training
trainer.train()

    # Step 8: Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Training complete and model saved!")