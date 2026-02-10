from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import base64
import os
import io
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='public')
CORS(app)  

MODEL_PATH = 'best_traffic_sign_model.keras'  
LABELS_PATH = 'labels.csv'

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Could not load model: {str(e)}")
    model = None

# Load labels
try:
    labels_df = pd.read_csv(LABELS_PATH)
    class_names = labels_df['Name'].tolist()
    logger.info(f"Loaded {len(class_names)} label classes")
except Exception as e:
    logger.error(f"Could not load labels: {str(e)}")
    class_names = []

def preprocess_image(image, target_size=(32, 32)):
    """Preprocess the image for model prediction."""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize(target_size)
        
        image_array = np.array(image) / 255.0  # Normalize to [0,1]
        
        processed_image = np.expand_dims(image_array, axis=0)
        
        logger.info(f"Image preprocessed successfully. Shape: {processed_image.shape}")
        return processed_image
    except Exception as e:
        logger.error(f"Error in preprocessing image: {str(e)}")
        raise

def enhance_image_for_detection(image):
    """Apply advanced enhancements specifically for traffic sign detection."""
    try:
        
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        original_height, original_width = img_cv.shape[:2]
        logger.info(f"Original image dimensions: {original_width}x{original_height}")
        
        img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)
        
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:,:,2] = clahe.apply(hsv[:,:,2])
        
        # Convert back to BGR
        img_cv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        edges = cv2.Canny(img_cv, 100, 200)
        
        img_cv = cv2.edgePreservingFilter(img_cv, flags=1, sigma_s=60, sigma_r=0.4)
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        detected_circles = None
        sign_detected = False
        
        if min(original_width, original_height) > 100:
            try:
                detected_circles = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                    param1=50, param2=30, minRadius=10, maxRadius=min(original_width, original_height) // 3
                )
            except Exception as circle_err:
                logger.warning(f"Circle detection failed: {str(circle_err)}")
        
        if detected_circles is not None and len(detected_circles) > 0:
            detected_circles = np.uint16(np.around(detected_circles))
            
            for i in detected_circles[0, :1]:  # Only process the first circle
                center_x, center_y, radius = i[0], i[1], i[2]
                
                padding = int(radius * 0.2)  # 20% padding
                x1 = max(0, center_x - radius - padding)
                y1 = max(0, center_y - radius - padding)
                x2 = min(original_width, center_x + radius + padding)
                y2 = min(original_height, center_y + radius + padding)
                
                # Crop to the region of interest if it's a reasonable size
                if (x2 - x1) > 30 and (y2 - y1) > 30:
                    img_cv = img_cv[y1:y2, x1:x2]
                    sign_detected = True
                    logger.info(f"Traffic sign candidate detected and cropped to: {x2-x1}x{y2-y1}")
        
       
        if not sign_detected:
            logger.info("No circular signs detected, attempting color-based detection")
            
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])
            
            # Blue range
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # Create masks for each color
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Combine masks
            mask = cv2.bitwise_or(mask_red1, mask_red2)
            mask = cv2.bitwise_or(mask, mask_blue)
            
            # Find contours in the combined mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
           
            if contours:
                # Get the largest contour by area
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                min_area = (original_width * original_height) * 0.01 
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    padding = int(min(w, h) * 0.2)  # 20% padding
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(original_width, x + w + padding)
                    y2 = min(original_height, y + h + padding)
                    
                    img_cv = img_cv[y1:y2, x1:x2]
                    sign_detected = True
                    logger.info(f"Traffic sign candidate detected through color segmentation: {x2-x1}x{y2-y1}")
    
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_cv = cv2.filter2D(img_cv, -1, kernel)
        
        b, g, r = cv2.split(img_cv)

        for channel in [b, g, r]:
            channel[:] = clahe.apply(channel)

        img_cv = cv2.merge([b, g, r])
        
        # Convert back to PIL format
        enhanced_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        logger.info("Image enhancement applied successfully")
        return enhanced_image
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        return image

def make_prediction(processed_image):
    """Make prediction with confidence scoring and validation."""
    try:
        # Make prediction
        logger.info("Making prediction with model")
        predictions = model.predict(processed_image)
        
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [(i, float(predictions[0][i])) for i in top_indices]
        
        # Log top predictions
        for idx, (class_idx, confidence) in enumerate(top_predictions):
            logger.info(f"Top {idx+1} prediction: class={class_idx}, name={class_names[class_idx]}, confidence={confidence:.4f}")
        
        # Get primary prediction
        predicted_class = top_predictions[0][0]
        confidence = top_predictions[0][1]
        
        if confidence < 0.4: 
            logger.warning(f"Low confidence prediction: {confidence:.4f}")
        
        # Get class name
        class_name = class_names[predicted_class]
        
        return {
            "class_id": int(predicted_class),
            "class_name": class_name,
            "confidence": confidence,
            "top_predictions": [
                {"class_id": int(class_id), "class_name": class_names[class_id], "confidence": conf}
                for class_id, conf in top_predictions
            ]
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route('/api/detect', methods=['POST'])
def detect_sign():
    if 'image' not in request.files and 'image_data' not in request.json:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        if 'image' in request.files:
            # Handle file upload
            logger.info("Processing uploaded file image")
            image_file = request.files['image']
            img = Image.open(image_file)
        else:
            # Handle base64 encoded image
            logger.info("Processing base64 image data")
            image_data = request.json['image_data']
            # Remove the data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64 data
            try:
                image_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(image_bytes))
                logger.info(f"Decoded base64 image: size={img.size}, mode={img.mode}")
            except Exception as decode_error:
                logger.error(f"Error decoding base64 image: {str(decode_error)}")
                return jsonify({"error": f"Invalid image data: {str(decode_error)}"}), 400
            
            # Apply enhancement for camera captured images
            img = enhance_image_for_detection(img)
        
        # Preprocess the image
        processed_image = preprocess_image(img)
        
        # Make prediction
        result = make_prediction(processed_image)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in detect_sign: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/signs', methods=['GET'])
def get_signs():
    """Return all available traffic signs."""
    signs = []
    for i, name in enumerate(class_names):
        signs.append({"id": i, "name": name})
    return jsonify(signs)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Server is running"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7702))
    app.run(host='0.0.0.0', port=port, debug=True)
