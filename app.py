from predict import predict_one
from model import FraudDetector
from flask import Flask, render_template, jsonify
import pickle
from pymongo import MongoClient
app = Flask(__name__)

fd = None
coll = None

json_fields = ['text_desc', 'fraud_predicted', 'fraud_probability']
def format_for_json(doc):
    return {
        f: doc[f] for f in json_fields if f in doc
    }

# home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# predict one case
@app.route('/score', methods=['POST'])
def score():
    prediction = predict_one(fd, coll)
    return jsonify(format_for_json(prediction))

# dashboard page
@app.route('/dashboard', methods=['GET'])
def dashboard():
    cursor = coll.find({})
    results = []
    for doc in cursor:
        results.append(format_for_json(doc))
    return jsonify(results)

if __name__ == '__main__':
    fd = pickle.load(open('data/model.pkl', 'rb'))

    mongo_client = MongoClient()
    db = mongo_client.mydb2
    coll = db.fraud_predictions
    app.run(host='0.0.0.0', port=80, threaded=True)
    # app.run(host='0.0.0.0', port=8080, debug=True)
