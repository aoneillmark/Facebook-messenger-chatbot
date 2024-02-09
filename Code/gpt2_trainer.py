from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

####################################################
# Functions

def train_loop(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs=3):
    for epoch in range(epochs):
        print("\n\nEpoch", epoch+1)
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_dataloader):
            # Print progress
            if i % 10 == 0 and not i == 0:
                print(f"Batch {i} of {len(train_dataloader)} | Average loss: {total_loss / i}")
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_labels = b_labels.to(device)
            
            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        # After training loop for each epoch, add validation evaluation
        model.eval()  # Evaluation mode
        val_total_loss = 0

        for batch in val_dataloader:
            b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
            
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            
            val_total_loss += outputs.loss.item()

        avg_val_loss = val_total_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss}")

        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Average training loss: {avg_train_loss}")



def encode_data(prompts, responses, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []
    labels = []

    for prompt, response in zip(prompts, responses):
        combined_text = prompt + tokenizer.eos_token + response  # Use eos_token as a separator
        encoded_dict = tokenizer(combined_text, add_special_tokens=True, 
                                 max_length=max_length, truncation=True, 
                                 padding='max_length', return_tensors='pt')

        # Extracting tensors from the encoded dictionary
        input_id = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']

        # Creating labels: tokens corresponding to `prompt + eos_token` are -100
        label = input_id.clone()
        label[label == tokenizer.pad_token_id] = -100  # Ignore pad tokens by setting their labels to -100
        sep_token_index = (input_id == tokenizer.eos_token_id).nonzero(as_tuple=True)[1]
        if sep_token_index.shape[0] > 1:  # If eos_token appears more than once, set the first part as -100
            label[:, :sep_token_index[0] + 1] = -100

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(label)

    # Concatenating lists of tensors along the batch dimension
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.cat(labels, dim=0)

    return input_ids, attention_masks, labels

####################################################
# Importing dependencies and checking for GPU

# Load pre-trained model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Check and print if GPU is available
print("CUDA available: ", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


####################################################
# Dataset Preprocessing

dataset_path = 'Results/prompt_response_pairs.txt'
prompts = []
responses = []

with open(dataset_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split(' [RESPONSE] ')
        if len(parts) == 2:
            prompt, response = parts
            prompt = prompt.replace('[PROMPT] ', '').strip()
            response = response.strip()
            prompts.append(prompt)
            responses.append(response)

# Split the data into training and validation sets
train_prompts, val_prompts, train_responses, val_responses = train_test_split(prompts, responses, test_size=0.1, random_state=42)

# Tokenize and encode the prompts and responses
max_length = 128  # (You can adjust based on your dataset and memory constraints, this worked dandy for me)
# Apply the function
train_input_ids, train_attention_masks, train_labels = encode_data(train_prompts, train_responses, tokenizer, max_length)
val_input_ids, val_attention_masks, val_labels = encode_data(val_prompts, val_responses, tokenizer, max_length)

# Dataset and DataLoader Preparation
train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=8)

val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=8)

####################################################
# Training Setup

# Example adjustment
optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)  # Adjust learning rate
# Training Loop
epochs = 1
# Learning rate scheduling
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*epochs)

# Train the model
train_loop(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs)

####################################################
# Save the model and tokenizer

model_save_path = 'Results/gpt2-finetuned'
tokenizer_save_path = 'Results/gpt2-finetuned'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)
