from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# Name des Chatbots definieren
CHATBOT_NAME = "Fenrys"

# Überprüfen, ob eine GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modell und Tokenizer laden
model = GPTNeoForCausalLM.from_pretrained('./finetuned_gpt_neo').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('./finetuned_gpt_neo')

# Funktion zum Generieren von Antworten
def generate_response(input_text):
    # Erstelle ein spezifisches Prompt, das den Chatbot anweist, auf die Anfrage zu antworten
    prompt = f"{CHATBOT_NAME}, hier ist eine Frage: {input_text} Antworte in einer natürlichen und hilfreichen Weise."
    
    # Tokenisierung des Eingangs
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=150)
    
    # Übertrage die Eingabedaten auf die GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generiere die Antwort mit attention_mask
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150, num_return_sequences=1)
    
    # Dekodiere die generierte Antwort
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extrahiere die Antwort ohne den Prompt
    response = response.replace(prompt, "").strip()  # Entferne den Prompt-Teil aus der Antwort
    response_with_name = f"{CHATBOT_NAME}: {response}"
    return response_with_name

# Benutzerinteraktion
print(f"{CHATBOT_NAME} ist bereit! Tippe 'exit', um zu beenden.")
while True:
    user_input = input("Du: ")
    if user_input.lower() == 'exit':
        print(f"{CHATBOT_NAME}: Auf Wiedersehen!")
        break
    response = generate_response(user_input)
    print(response)
