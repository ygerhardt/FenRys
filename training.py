import json
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import numpy as np
from torch.cuda.amp import GradScaler, autocast

# 1. Datensatz laden
with open('data/chat_data.json', 'r') as f:
    data = json.load(f)

# 2. Erstellen eines Datasets
dataset = Dataset.from_dict({
    'input': [entry['input'] for entry in data],
    'response': [entry['response'] for entry in data]
})

# 3. Tokenizer und Modell laden
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
tokenizer.pad_token = tokenizer.eos_token  # Verwende das End-of-Sequence-Token als Padding-Token
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M', pad_token_id=tokenizer.eos_token_id)

# 4. Funktion zum Tokenisieren der Eingabedaten
def tokenize_function(examples):
    # Kombiniere Eingabe und Antwort für den Tokenizer
    texts = [f"{input_text} {tokenizer.eos_token} {response}" for input_text, response in zip(examples['input'], examples['response'])]
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors="pt", return_attention_mask=True)

# 5. Dataset tokenisieren
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 6. Labels hinzufügen
tokenized_datasets = tokenized_datasets.map(lambda x: {'labels': x['input_ids']}, batched=True)

# 7. Dataset für PyTorch formatieren
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 8. Datensatz aufteilen in Train- und Eval-Datensatz
train_size = int(0.8 * len(tokenized_datasets))  # 80% für das Training
train_dataset = tokenized_datasets.select(range(train_size))
eval_dataset = tokenized_datasets.select(range(train_size, len(tokenized_datasets)))

# 9. Funktion zur Berechnung der Perplexity (Evaluationsmetrik)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Wir berechnen den Loss basierend auf den Logits und Labels
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    perplexity = torch.exp(loss).item()
    return {"perplexity": perplexity}

# 10. Trainingsargumente festlegen
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Beispiel für Gradient Accumulation
    num_train_epochs=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,  # Lade das beste Modell am Ende
    metric_for_best_model='eval_loss',  # Metrik für das beste Modell
)

# 11. Trainer-Instanz erstellen
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# 12. Trainiere das Modell mit mixed precision
scaler = GradScaler()  # Initialisiere den GradScaler für mixed precision

with autocast():
    trainer.train()

# 13. Modell speichern
model.save_pretrained('./finetuned_gpt_neo')
tokenizer.save_pretrained('./finetuned_gpt_neo')
