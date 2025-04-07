from flask import Flask, request

app = Flask(__name__)

from transformers import pipeline
pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis", framework="pt")

@app.route('/analyze', methods=['POST'])
def analyze():
    # Try to get sentence from form (Twilio will send 'Body' for SMS/WhatsApp)
    sentence = request.form.get("Body") or request.json.get("sentence") if request.is_json else None

    if not sentence:
        return "No sentence provided", 400

    result = pipe(sentence)
    label = result[0]["label"]

    return label  # Just return label as string

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
