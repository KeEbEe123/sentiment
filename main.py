from flask import Flask, request, Response
from transformers import pipeline

app = Flask(__name__)

pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis", framework="pt")

@app.route('/analyze', methods=['POST'])
def analyze():
    sentence = request.form.get("Body")  # Twilio sends the WhatsApp text in "Body"

    if not sentence:
        return "No sentence provided", 400

    result = pipe(sentence)
    label = result[0]["label"]

    # Respond with TwiML (Twilio Markup Language) to send message back
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>Sentiment: {label}</Message>
</Response>"""

    return Response(twiml, mimetype="application/xml")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
