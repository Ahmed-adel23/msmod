from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from flask_restful import Resource 


model = load_model('cnn_model.h5')

input_shape = (150, 150, 3)

def preprocess_and_predict(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (input_shape[1], input_shape[0]))
        image = image / 255.0  
        prediction = model.predict(np.expand_dims(image, axis=0))
        class_label = np.argmax(prediction)

        class_name = get_code(class_label)

        accuracy = float(np.max(prediction))

        return class_name, accuracy

    except Exception as e:
        return str(e), 500

def get_code(n):
    code = {'Control-Axial': 0, 'Control-Sagittal': 1, 'MS-Axial': 2, 'MS-Sagittal': 3}
    for x, y in code.items():
        if n == y:
            return x
def image_to_base64(image):
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")  
    
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
def preprocess_image(image):
    try:
        image_pil = Image.open(image)
        
        image_pil = image_pil.resize((128, 128))
        
        image_pil = image_pil.convert('L')
        
        image_np = np.array(image_pil)
        image_np = image_np / 255.0  
        
        image_np = np.expand_dims(image_np, axis=-1)
        image_np = np.concatenate([image_np, image_np], axis=-1)
        
        return image_np

    except Exception as e:
        return str(e)
class MSPrediction3(Resource):
    def post(self):
        try:
            image = request.files['image']
            predicted_class, accuracy = preprocess_and_predict(image)
            image_data = preprocess_image(image)
            image_data = image_to_base64(image_data)
            return {'class_label':predicted_class , 'accuracy':float(accuracy), 'image_data':image_data}

        except Exception as e:
            return str(e)