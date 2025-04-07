from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the classification pipeline using PyTorch
pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis", framework="pt")

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    sentence = data.get("sentence")

    if not sentence:
        return "No sentence provided", 400

    result = pipe(sentence)
    label = result[0]["label"]

    return label  # Return only the label as a string

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
