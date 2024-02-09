from transformers import GPT2Tokenizer, GPT2LMHeadModel


def generate_response(prompt, max_length=100, use_max_new_tokens=True):
    encoding = tokenizer.encode_plus(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']
    
    model.eval()
    
    if use_max_new_tokens:
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            # max_length=None,  # Allows for flexible generation length, max_new_tokens takes precedence
            max_new_tokens=50,  # Increased for longer responses
            temperature=0.9,  # Typically 0.5-1.0. Higher gives more creative responses
            top_k=50,
            top_p=0.92,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            num_beams=5,
            no_repeat_ngram_size=3,  # Increased to reduce repetition
            early_stopping=True
        )
    else:
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=0.9,
            top_k=30,
            top_p=0.92,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# Load the trained model and tokenizer
model_path = 'Results/gpt2-finetuned'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)


# Example usage
prompt = "Hey there! "
response = generate_response(prompt, max_length=300)
print("Prompt:", prompt)
print("Response:", response)
