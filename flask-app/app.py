import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from text_preprocessor import preprocess_text

app = Flask(__name__)


def get_label_prediction(model, X_test):
    # get probabilities
    probabilities = model.predict_proba(X_test)

    # get best fitting label
    best_n = np.argsort(probabilities, axis=1)[:, -1:]

    # get label prediction
    predictions = [[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]

    predictions = [item[::-1] for item in predictions]

    return predictions


loaded_model = pickle.load(open('model_tfidf_xgb.pkl', 'rb'))
loaded_transformer = pickle.load(open('transformer_tfidf_xgb.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['article']
    predictor = loaded_transformer.transform([preprocess_text(text)])
    prediction = get_label_prediction(loaded_model, predictor)
    if prediction == [['real']]:
        prediction = 'real'
    else:
        prediction = 'false'

    return render_template('index.html', prediction_text=' Prediction: {}'.format(prediction))


# @app.route('/results', methods=['POST'])
# def results():
#     data = request.get_json(force=True)
#     predictor = loaded_transformer.transform(["President Trump AND THE impeachment story !!!"])
#     prediction = get_label_prediction(loaded_model, predictor)
#     output = prediction
#     return jsonify(output)


if __name__ == "__main__":
    app.secret_key = 'superSecretKey'
    app.run(debug=True, host='0.0.0.0',port=5000)
