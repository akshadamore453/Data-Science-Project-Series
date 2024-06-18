from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and vectorizer from the pickle files
with open('best_svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Mapping of class indices to sentiment labels
class_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        print(f"Received message: {message}")  # Debugging

        # Transform the input message using the TF-IDF vectorizer
        data = [message]
        vect = vectorizer.transform(data)
        print(f"Transformed vector: {vect}")  # Debugging

        # Predict probabilities
        prediction = model.predict_proba(vect)
        print(f"Model prediction probabilities: {prediction}")  # Debugging

        # Get the class index with the highest probability
        class_index = np.argmax(prediction, axis=1)[0]
        prediction_label = class_mapping[class_index]
        print(f"Prediction label: {prediction_label}")  # Debugging

    
        return render_template('index.html', prediction=prediction_label)

if __name__ == "__main__":
    app.run(debug=True)


