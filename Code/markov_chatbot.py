import json
import random
import re

# Build a Markov chain model from the input text
def build_markov_chain(text, chain={}):
    words = text.split()
    index = 1

    for word in words[index:]:
        key = words[index - 1]
        if key in chain:
            chain[key].append(word)
        else:
            chain[key] = [word]
        index += 1

    return chain

# Generate responses using the Markov chain
def generate_response(chain, prompt, count=10):
    words = prompt.split()
    start_word = words[-1] if words[-1] in chain else random.choice(list(chain.keys()))
    message = start_word.capitalize()

    for i in range(count-1):
        next_words = chain.get(start_word)
        if next_words:
            next_word = random.choice(next_words)
            message += ' ' + next_word
            start_word = next_word
        else:
            break  # Break the loop if no next word is found

    return message

# Clean the text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters or handle them as needed
    text = re.sub(r'[^\w\s\.\,\!\?\'\"]', '', text)
    return text

####################################################

# Load the JSON data from a file
with open('Data/message_1.json', 'r') as file:  # Correct the path as necessary
    chat_data = json.load(file)

# Clean and preprocess the user's messages
user_messages = ' '.join([msg['content'] for msg in chat_data['messages'] if msg.get('sender_name') == 'REPLACE_WITH_USER_NAME' and 'content' in msg])
user_messages_clean = clean_text(user_messages)

# Build the Markov chain from clean messages
chain = build_markov_chain(user_messages_clean)

# Input prompts and generate responses
print("/n")
prompt = "Hi there! How are you today?"
response = generate_response(chain, prompt)
print("Prompt: " + prompt)
print("Response: " + response)
