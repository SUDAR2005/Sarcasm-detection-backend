from flask import Flask,request,jsonify
from transformers import DistilBertTokenizer,DistilBertForSequenceClassification
from flask_cors import CORS
import torch

app=Flask(__name__)
CORS(app)
model_path="fine_tuned_distilbert_sarcasm"
tokenizer=DistilBertTokenizer.from_pretrained(model_path)
model=DistilBertForSequenceClassification.from_pretrained(model_path)

@app.route('/predict',methods=['POST'])
def predict():
    data=request.get_json(force=True)
    text=data['text']
    text=text.lower()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' key in request data"}),400
    inputs=tokenizer(text,return_tensors='pt',truncation=True,padding=True,max_length=128)
    
    with torch.no_grad():
        outputs=model(**inputs)
        logits=outputs.logits
        prediction=torch.argmax(logits,dim=1).item()
    
    response={
        'text': text,
        'is_sarcastic': bool(prediction)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)