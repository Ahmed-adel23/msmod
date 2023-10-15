from flask import Flask, request,  jsonify
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from io import BytesIO
import base64
from flask_restful import Resource, Api

def preprocess_image(image):
    
    image = image.resize((128, 128))
    image = image.convert('L')
    image = np.array(image)
    image = image / 255.0  
    image = np.expand_dims(image, axis=-1)
    image = np.concatenate([image, image], axis=-1)

    return image

def postprocess_mask(mask):
    threshold = 0.5
    mask = (mask > threshold).astype(np.uint8)
    return mask

def image_to_base64(image):
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")  
    
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
def dice_coef(y_true, y_pred, smooth=1.0):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)


def precision(y_true, y_pred):
    true_positives = tf.reduce_sum(y_true * tf.round(y_pred))
    predicted_positives = tf.reduce_sum(tf.round(y_pred))
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())


def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(y_true * tf.round(y_pred))
    possible_positives = tf.reduce_sum(y_true)
    return true_positives / (possible_positives + tf.keras.backend.epsilon())


def specificity(y_true, y_pred):
    true_negatives = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    possible_negatives = tf.reduce_sum(1 - y_true)
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())

model = tf.keras.models.load_model('D__segmentaion_model.h5', custom_objects={
    'accuracy': tf.keras.metrics.MeanIoU(num_classes=2),
    'dice_coef': dice_coef,
    'precision': precision,
    'sensitivity': sensitivity,
    'specificity': specificity
}, compile=False)

class MSPrediction4(Resource):
    def post(self):
        if request.method == 'POST':
            
            uploaded_file = request.files['file']

            if uploaded_file.filename != '':
                
                image = Image.open(uploaded_file)
                image = preprocess_image(image)

                
                segmentation_mask = model.predict(np.expand_dims(image, axis=0))[0]

                
                segmentation_mask = postprocess_mask(segmentation_mask)

                
                original_image_base64 = image_to_base64(image)
                mask_image_base64 = image_to_base64(segmentation_mask)

                return {'original_image':original_image_base64, 'mask_image':mask_image_base64}

        return jsonify(error="No file uploaded")

