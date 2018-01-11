import pandas as pd
import json

def predict_one(fd, coll):
    X = pd.read_json('http://galvanize-case-study-on-fraud.herokuapp.com/data_point', typ='series')
    X = pd.DataFrame([X])
    class_prediction = fd.predict(X)[0]
    proba_prediction = fd.predict_proba(X)[0][1]
    X['fraud_predicted'] = class_prediction
    X['fraud_probability'] = proba_prediction
    record = json.loads(X.T.to_json()).values()
    coll.insert(record)
    return json.loads(X.iloc[0].to_json())
