from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
import torch
import json
import os
import numpy as np
from datasets import Dataset
from collections import Counter

# Name des Chatbots definieren
CHATBOT_NAME = "Fenrys"

# Überprüfen, ob eine GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modell und Tokenizer laden
model = GPTNeoForCausalLM.from_pretrained('./finetuned_gpt_neo').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('./finetuned_gpt_neo')

# Funktion zum Generieren von Antworten
def generate_response(input_text):
    prompt = f"Frage: {input_text} Antwort:"
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=100)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_length=100,
        num_return_sequences=1, 
        temperature=0.7,  
        top_p=0.9,  
        repetition_penalty=1.1,  
        do_sample=True  
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()  
    response_with_name = f"{CHATBOT_NAME}: {response}"
    return response_with_name

# Funktion zur Speicherung neuer Konversationsdaten
def save_conversation(user_input, bot_response, feedback, correct_answer=None, file_path='data/chat_data.json'):
    new_entry = {
        'input': user_input,
        'response': bot_response,
        'feedback': feedback
    }
    
    if feedback == 'nein' and correct_answer:  # Speichere die richtige Antwort, wenn Feedback negativ ist
        new_entry['correct_answer'] = correct_answer

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    data.append(new_entry)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Funktion zur Analyse von häufigen Missverständnissen
def analyze_feedback(training_data_path='data/chat_data.json'):
    try:
        with open(training_data_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Keine Trainingsdaten gefunden.")
        return

    # Überprüfen, ob das 'feedback'-Feld existiert
    incorrect_responses = [entry for entry in data if entry.get('feedback') == 'nein']
    print(f"Anzahl falscher Antworten: {len(incorrect_responses)}")

    if not incorrect_responses:
        return

    inputs = [entry['input'] for entry in incorrect_responses]
    input_counter = Counter(inputs)
    
    print("Häufigste Missverständnisse:")
    for input_text, count in input_counter.most_common(5):
        print(f"'{input_text}' - {count} Mal falsch beantwortet")

# Funktion zur Training des Modells mit neuen Daten
def retrain_model(training_data_path='data/chat_data.json', min_feedback=10):
    try:
        with open(training_data_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Keine Trainingsdaten gefunden.")
        return

    if len(data) < min_feedback:
        print(f"Nicht genügend Feedback für das Training. Benötigt: {min_feedback}, Vorhanden: {len(data)}")
        return

    analyze_feedback(training_data_path)
    
    # Dataset für das Training vorbereiten
    train_dataset = Dataset.from_dict({
        'input': [entry['input'] for entry in data],
        'response': [entry['correct_answer'] if 'correct_answer' in entry else entry['response'] for entry in data]
    })

    def tokenize_function(examples):
        texts = [f"{input_text} {tokenizer.eos_token} {response}" for input_text, response in zip(examples['input'], examples['response'])]
        return tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors="pt", return_attention_mask=True)

    tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.map(lambda x: {'labels': x['input_ids']}, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        evaluation_strategy='steps',
        eval_steps=500,
        save_steps=500,
        learning_rate=5e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    trainer.train()
    
    model.save_pretrained('./finetuned_gpt_neo')
    tokenizer.save_pretrained('./finetuned_gpt_neo')
    print("Modell erfolgreich neu trainiert und gespeichert.")

# Benutzerinteraktion
print(f"{CHATBOT_NAME} ist bereit!")
while True:
    user_input = input("Du: ")
    if user_input.lower() == 'bye':
        print(f"{CHATBOT_NAME}: Auf Wiedersehen!")
        break
    response = generate_response(user_input)
    print(response)

    feedback = input("War diese Antwort korrekt? (ja/nein): ").strip().lower()
    if feedback not in ['ja', 'nein']:
        print("Bitte antworte mit 'ja' oder 'nein'.")
        continue

    correct_answer = None
    if feedback == 'nein':
        correct_answer = input("Was wäre die richtige Antwort?: ").strip()  # Benutzer kann die richtige Antwort angeben

    save_conversation(user_input, response, feedback, correct_answer)

    if len(json.load(open('data/chat_data.json'))) % 10 == 0:
        retrain_model()
