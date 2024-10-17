import json
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch

# 1. GermanQuAD-Datensatz laden
dataset = load_dataset('deepset/germanquad', split='train', trust_remote_code=True)

# 2. Erstellen eines Datasets
# Nutze Fragen und die erste Antwort aus jeder Antwortliste als Eingabe
qa_dataset = Dataset.from_dict({
    'input': dataset['question'],  # Fragen als Eingabe
    # Wähle die erste Antwort aus der Liste der Antworten
    'response': [answers['text'][0] if answers['text'] else "" for answers in dataset['answers']]  # Erste Antwort als Antwort
})

# 3. Tokenizer und Modell laden
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
tokenizer.pad_token = tokenizer.eos_token  # Verwende das End-of-Sequence-Token als Padding-Token
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M', pad_token_id=tokenizer.eos_token_id)

# 4. Funktion zum Tokenisieren der Eingabedaten
def tokenize_function(examples):
    # Kombiniere Eingabe und Antwort für den Tokenizer
    texts = [f"Frage: {input_text} {tokenizer.eos_token} Antwort: {response}" for input_text, response in zip(examples['input'], examples['response'])]
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors="pt", return_attention_mask=True)

# 5. Dataset tokenisieren
tokenized_datasets = qa_dataset.map(tokenize_function, batched=True)

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
    eval_strategy='steps',  # Verwende 'steps' für die Evaluation
    eval_steps=1000,  # Evaluation alle 1000 Schritte
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=10,  # Reduziert auf 10 Epochen für den Testlauf
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
    save_strategy="steps",  # Speichern alle 1000 Schritte
    save_steps=1000,  # Speichern alle 1000 Schritte
    load_best_model_at_end=True,  # Lade das beste Modell am Ende
    metric_for_best_model="perplexity",  # Die Metrik für das beste Modell
    greater_is_better=False,  # Bei Perplexity ist weniger besser
)

# 11. Trainer-Instanz erstellen
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# 12. Trainiere das Modell
trainer.train()  # Trainiere für alle Epochen

# 13. Modell speichern
model.save_pretrained('./finetuned_gpt_neo')
tokenizer.save_pretrained('./finetuned_gpt_neo')
