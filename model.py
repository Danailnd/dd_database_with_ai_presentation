# model.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the tokenizer and pre-trained model
model_name = "distilgpt2"  # Use a pre-trained model like GPT-2
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add the padding token to the tokenizer, needed for GPT-2
tokenizer.pad_token = tokenizer.eos_token


# Generate response based on fine-tuned model
def generate_response(message, mode="pretrained"):
    if mode == "fine-tuned":
        model, tokenizer = load_fine_tuned_model()
    else:
        model, tokenizer = GPT2LMHeadModel.from_pretrained("distilgpt2"), GPT2Tokenizer.from_pretrained("distilgpt2")

    inputs = tokenizer.encode(message, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, temperature=1.2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Function to load the fine-tuned model
def load_fine_tuned_model():
    model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")
    tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")
    return model, tokenizer
