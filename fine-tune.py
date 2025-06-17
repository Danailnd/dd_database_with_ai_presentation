# fine_tune.py
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import mysql.connector

# Database connection configuration
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


# Step 2: Fetch data from the database
def get_training_data_from_db():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT prompt, response FROM training_data")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []


# Prepare the data for fine-tuning
def prepare_fine_tuning_data():
    data = get_training_data_from_db()

    if not data:
        raise Exception("No data found in the database for training.")

    # Process the data into a format that can be used for fine-tuning
    training_data = []
    for entry in data:
        prompt = entry['prompt']
        response = entry['response']
        training_data.append({'text': f"{prompt} {tokenizer.eos_token} {response}"})

    return Dataset.from_list(training_data)


# Fine-tune the model
def fine_tune_model():
    dataset = prepare_fine_tuning_data()

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

    print("Fine-tuning complete and model saved!")

