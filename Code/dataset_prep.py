import json
import re
import os

# Function to clean and preprocess text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters or handle them as needed
    text = re.sub(r'[^\w\s\.\,\!\?\'\"]', '', text)
    return text

# Define the function to process each file
def process_chat_data(file_path, user_name):
    # Load the chat data
    with open(file_path, 'r', encoding='utf-8') as f:
        chat_data = json.load(f)
    
    # Sort messages by timestamp to ensure the conversation flow is maintained
    messages = sorted(chat_data['messages'], key=lambda x: x['timestamp_ms'])
    
    # Create prompt-response pairs where the user of interest is is the response sender
    prompt_response_pairs = []
    for i in range(1, len(messages)):  # Start from 1 since we look back for the prompt
        if messages[i].get('sender_name') == user_name and 'content' in messages[i]:
            # Use the previous message as the prompt
            if 'content' in messages[i-1]:
                prompt = clean_text(messages[i-1]['content'])
                response = clean_text(messages[i]['content'])
                pair = f"[PROMPT] {prompt} [RESPONSE] {response}"
                prompt_response_pairs.append(pair)
    
    return prompt_response_pairs

# Get the list of all json files in the Data directory
data_directory = 'Data/'
json_files = [f for f in os.listdir(data_directory) if f.endswith('.json')]

# User of interest, this is who the bot will emulate
user_name = "Arthur Dent"

# Process all files and aggregate prompt-response pairs
all_pairs = []
for json_file in json_files:
    file_path = os.path.join(data_directory, json_file)
    all_pairs.extend(process_chat_data(file_path, user_name))

# Save all prompt-response pairs to a single file
output_file = 'Results/prompt_response_pairs.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for pair in all_pairs:
        f.write(f"{pair}\n")
