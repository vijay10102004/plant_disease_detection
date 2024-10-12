import os
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

model = tf.keras.models.load_model('plant_disease_model.keras')

class_labels = [
    'Apple Healthy',
    'Sugarcane Yellow Leaf',
    'Tomato Bacterial Spot',
    'Tomato Early Blight',
    'Tomato Healthy',
    'Tomato Late Blight',
    'Tomato Leaf Curl',
    'Tomato Leaf Mold',
    'Tomato Mosaic Virus',
    'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites',
    'Tomato Target Spot'
]

recommendations = {
    'Apple Healthy': ["1.Ensure proper irrigation", "2.Maintain pest control"],
    'Sugarcane Yellow Leaf': ["1.Use fungicides to control leaf spot.", "2.Provide adequate nitrogen fertilizers."],
    'Tomato Bacterial Spot': ["1.Apply copper-based bactericides.", "2.Ensure crop rotation to prevent spread."],
    'Tomato Early Blight': ["1.Use fungicide to control early blight.", "2.Remove and destroy infected leaves."],
    'Tomato Healthy': ["1.Continue regular care and monitoring.", "2.Ensure proper sunlight and water levels."],
    'Tomato Late Blight': ["1.Use fungicide to control late blight.", "2.Avoid overhead watering."],
    'Tomato Leaf Curl': ["1.Control whitefly population.", "2.Use disease-free transplants."],
    'Tomato Leaf Mold': ["1.Improve air circulation around plants.", "2.Use fungicide to prevent mold growth."],
    'Tomato Mosaic Virus': ["1.Remove infected plants immediately.", "2.Sanitize tools between plant handling."],
    'Tomato Septoria Leaf Spot': ["1.Apply fungicides regularly.", "2.Remove lower leaves to reduce spread."],
    'Tomato Spider Mites': ["1.Use insecticidal soap or oil.", "2.Introduce natural predators like ladybugs."],
    'Tomato Target Spot': ["1.Use a fungicide that targets this disease.", "2.Remove infected leaves to reduce spread."]
}

def predict(image_path):
    img = Image.open(image_path).resize((256, 256)) 
    img_array = np.array(img).astype(np.float32)  
    img_array = img_array / 255.0 
    img_array = img_array[np.newaxis, ...]  
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds, axis=1)[0] 
    predicted_label = class_labels[predicted_index]
    confidence = int(np.max(preds) * 100)
    disease_recommendations = recommendations.get(predicted_label, ["No recommendations available"])
    return predicted_label, confidence, disease_recommendations

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password': 
            session['logged_in'] = True
            return redirect(url_for('index')) 
        else:
            return "Invalid credentials. Please try again."

    return render_template('Login.html') 

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login')) 
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        today = datetime.now().strftime("%Y-%m-%d")
        upload_folder = os.path.join("static/uploads", today)
        os.makedirs(upload_folder, exist_ok=True)
        image_path = os.path.join(upload_folder, file.filename)
        file.save(image_path)
        
        predicted_label, confidence, disease_recommendations = predict(image_path)
        
        return render_template('index.html', prediction=predicted_label, confidence=confidence,
                               recommendations=disease_recommendations, image_path=image_path)
    
    return render_template('index.html', prediction=None)

@app.route('/logout')
def logout():
    session.pop('logged_in', None) 
    return redirect(url_for('login'))  

if __name__ == '__main__':
    app.run(debug=True)