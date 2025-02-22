from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
save_directory = "./models/opus-mt-en-ROMANCE"

# Download model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save them locally
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model saved in: {save_directory}")