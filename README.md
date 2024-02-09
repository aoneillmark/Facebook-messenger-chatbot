# Facebook Messenger ChatBot from User Data

This repository provides tools to import your Facebook Messenger data in JSON format, and train a bot to emulate one of the users in the group chat using GPT-2.

## Disclaimer

This is a small project that anyone can make in an afternoon using an old GPT-2 model. The responses are often very silly and nonsensical, and easily distinguished from real messages.
That said, users of this repository are expected to approach this responsibly, and never attempt to train a chatbot based on someone else without their permission, and users must respect the privacy of others.

I do not condone or support any use of this repository for malicious purposes, and in using this repository you accept all liability for your actions.

## Project Structure

- `Code/` - Contains all the Python scripts for data preparation and model training.
  - `dataset_prep.py` - Script for preparing the dataset from the JSON files.
  - `gpt2_prompter.py` - Script for generating responses using the GPT-2 model.
  - `gpt2_trainer.py` - Script for training the GPT-2 model.
  - `markov_chatbot.py` - Script for a Markov chain-based chatbot.
- `Data/` - Directory to place your JSON data files.
- `Results/` - Directory where the results of the model training will be stored.

## How to Use

1. Download your JSON messenger data from Facebook (https://www.facebook.com/help/messenger-app/677912386869109)
2. Place your Facebook Messenger data in JSON format in the `Data/` directory.
3. Run the `dataset_prep.py` script to prepare the data for training.
4. Run the `gpt2_trainer.py` script to train the GPT-2 model.
5. Use the `gpt2_prompter.py` script to generate responses from the trained model.
