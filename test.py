import json
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

class SimpleChatbot:
    def __init__(self):
        self.user_data = {}
        self.lemmatizer = WordNetLemmatizer()
        self.load_user_data()

    def load_user_data(self):
        try:
            with open('user_data.json', 'r') as file:
                self.user_data = json.load(file)
        except FileNotFoundError:
            self.user_data = {"default": {"history": [], "preferences": {}, "hobbies": []}}
        except json.JSONDecodeError:
            self.user_data = {"default": {"history": [], "preferences": {}, "hobbies": []}}

    def save_user_data(self):
        with open('user_data.json', 'w') as file:
            json.dump(self.user_data, file)

    def process_input(self, user_input):
        tokens = word_tokenize(user_input.lower())
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def respond(self, user_input):
        user_id = "default"
        processed_input = self.process_input(user_input)
        response = "Das ist interessant!"
        self.user_data[user_id]["history"].append(user_input)
        self.save_user_data()
        return response

    def chat(self):
        print("Hallo! Ich bin Fenrys, dein Chatbot. Wie kann ich dir helfen?")
        while True:
            user_input = input("Du: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Chatbot: Auf Wiedersehen!")
                break
            response = self.respond(user_input)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot = SimpleChatbot()
    chatbot.chat()
