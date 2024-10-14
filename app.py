from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# ดิกชันนารีสำหรับระบุคลาสที่เป็นไปได้ทั้งหมด (เหลือเพียง 4 คลาส)
dic = {0: 'angry', 1: 'happy', 2: 'relaxed', 3: 'sad'}

# โหลดโมเดลที่ฝึกมาแล้วสำหรับตรวจจับอารมณ์สุนัข
emotion_model = load_model('final_model85.h5')
emotion_model.make_predict_function()

# โหลดโมเดลที่ฝึกมาแล้วสำหรับตรวจสอบว่าสุนัขหรือไม่
dog_detector_model = load_model('keras_model.h5')
dog_detector_model.make_predict_function()

def is_dog(img):
    i = img.resize((224, 224))  # เปลี่ยนขนาดเป็น 224x224
    i = np.array(i) / 255.0
    i = np.expand_dims(i, axis=0)
    p = np.argmax(dog_detector_model.predict(i), axis=-1)
    
    # ตรวจสอบว่าเป็นสุนัขหรือไม่
    return p[0] == 0  # สมมติว่า 0 คือสุนัข, 1 คือไม่ใช่สุนัข

def predict_emotion(img):
    i = img.resize((224, 224))  # เปลี่ยนขนาดเป็น 224x224
    i = np.array(i) / 255.0
    i = np.expand_dims(i, axis=0)
    prediction = emotion_model.predict(i)[0]  # คาดการณ์ทั้งหมด
    predicted_class = np.argmax(prediction)  # คลาสที่มีความน่าจะเป็นสูงสุด
    predicted_probability = prediction[predicted_class] * 100  # คำนวณเปอร์เซ็นต์

    # แสดงผลการทำนาย (เฉพาะ 4 คลาส)
    return dic.get(predicted_class, 'unknown emotion'), predicted_probability

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def get_output():
    if 'my_image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    img_file = request.files['my_image']
    
    if img_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(img_file.stream)
    except Exception as e:
        return jsonify({"error": "Error opening image: " + str(e)}), 400
    
    try:
        if not is_dog(img):
            return jsonify({"prediction": "This is not a dog image"}), 200

        prediction, probability = predict_emotion(img)
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # แสดงผลการทำนาย โดยไม่มี 'undefined%' ที่ไม่ต้องการ
        return jsonify({
            "prediction": f"The dog looks {prediction} ({probability:.2f}%)",  # ใช้เปอร์เซ็นต์ที่คำนวณได้
            "img_data": "data:image/jpeg;base64," + img_data
        })
    except Exception as e:
        return jsonify({"error": "Error processing image: " + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
