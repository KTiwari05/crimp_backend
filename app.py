from flask import Flask, request, jsonify, make_response, send_file, abort
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
from datetime import timedelta
from models import db, User
from PIL import Image
import fitz
import io
import os
import tempfile
from werkzeug.utils import safe_join
import cv2
import numpy as np
import uuid
import sys
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import time  # new import
 
app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://10.245.146.250:5004"])
 
# Configuration settings
app.config.from_object("config.ApplicationConfig")
app.config['JWT_SECRET_KEY'] = app.config['SECRET_KEY']
app.config['JWT_COOKIE_SECURE'] = False
app.config['JWT_TOKEN_LOCATION'] = ['cookies']
app.config['JWT_ACCESS_COOKIE_PATH'] = '/'
app.config['JWT_ACCESS_COOKIE_NAME'] = 'access_token_cookie'  # Ensure consistency
app.config['JWT_COOKIE_CSRF_PROTECT'] = False  # Disable CSRF protection
 
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB file limit
 
# Disable CSRF protection for API routes (if using Flask-WTF or Flask-SeaSurf)
app.config['WTF_CSRF_ENABLED'] = False
 
jwt = JWTManager(app)
bcrypt = Bcrypt(app)
db.init_app(app)
 
with app.app_context():
    db.create_all()
 
# User registration route
@app.route("/register", methods=["POST"])
def register_user():
    data = request.json
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"message": "Email and password are required"}), 400
 
    user_exists = User.query.filter_by(email=data["email"]).first() is not None
    if user_exists:
        return jsonify({"message": "User with that email already exists"}), 409
 
    hashed_password = bcrypt.generate_password_hash(data["password"]).decode("utf-8")
    new_user = User(email=data["email"], password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
 
    return jsonify({"id": new_user.id, "email": new_user.email}), 201
 
# User login route
@app.route("/login", methods=["POST"])
def login_user():
    data = request.json
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"message": "Email and password are required"}), 400
 
    user = User.query.filter_by(email=data["email"]).first()
    if user is None or not bcrypt.check_password_hash(user.password, data["password"]):
        return jsonify({"message": "Invalid email or password"}), 401
 
    access_token = create_access_token(identity=user.id, expires_delta=timedelta(hours=1/2))
    response = make_response(jsonify({
        "message": "Login successful",
        "id": user.id,
        "email": user.email,
        "token": access_token
    }))
    response.set_cookie(
        'access_token_cookie',
        access_token,
        httponly=True,
        samesite='Lax',
        path='/'
    )
 
    return response, 200
 
# User logout route
@app.route("/logout", methods=["POST"])
@jwt_required()
def logout_user():
    response = make_response(jsonify({"message": "Logout successful"}))
    response.delete_cookie('access_token_cookie', path='/')
    return response, 200
 
# Get current user route
@app.route("/@me", methods=["GET"])
@jwt_required()
def get_current_user():
    user_id = get_jwt_identity()
    user = User.query.filter_by(id=user_id).first()
    if not user:
        return jsonify({"message": "User not found"}), 404
 
    return jsonify({"id": user.id, "email": user.email}), 200
 
 
# Flask endpoint to send the comparison result for download (optional)
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    print(f"Debug: /download route hit with filename={filename}")
    file_path = safe_join(tempfile.gettempdir(), filename)
    if not os.path.exists(file_path):
        abort(404)
    try:
        response = send_file(file_path, as_attachment=True, mimetype='application/pdf')
        print(f"Debug: Sending file {file_path}")
        # Schedule the file for deletion after sending
        @response.call_on_close
        def remove_file():
            try:
                os.unlink(file_path)
                print(f"Deleted temporary file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {str(e)}")
        return response
    except Exception as e:
        return jsonify({"error": f"Failed to download file: {str(e)}"}), 500

MODEL_PATH = os.path.join('model', 'crimp_defect_detector.pth')
loaded_model = None

def process_image(image_path, threshold=20):
    # Load image from disk
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"Error: Could not read image at {image_path}")
    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    filtered = cv2.GaussianBlur(filtered, (5, 5), 0)
    
    # Convert to HSV and create mask to segment colored regions
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    lower_bound = (0, 30, 50)   
    upper_bound = (179, 255, 255)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Morphological operations to remove small noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Adjusted kernel size
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Increased iterations
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # Increased iterations
    
    # Use contour hierarchy to extract outer and inner contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    outer_contours = []
    inner_contours = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, h in enumerate(hierarchy):
            area = cv2.contourArea(contours[i])
            if area < 100:
                continue
            # h[3] == -1: no parent, outer contour; else inner contour (hole)
            if h[3] == -1:
                outer_contours.append(contours[i])
            else:
                inner_contours.append(contours[i])
    
    # Refine inner contours: perform a color-difference check between inside and its rim
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_image = cv2.medianBlur(lab_image, 7)  # Increased blur size from 5 to 7 for better noise reduction
    # New: create a grayscale version for gap detection (from test4.py logic)
    gray_for_gap = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    refined_inner_contours = []
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))  # Adjusted kernel size
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    for cnt in inner_contours:
        mask_inner = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask_inner, [cnt], -1, 255, -1)
        # Erode the mask to isolate the inner area better
        eroded_mask = cv2.erode(mask_inner, kernel_erode, iterations=2)  # Increased iterations
        mean_inside = cv2.mean(lab_image, mask=eroded_mask)[:3]
        
        # Create a rim by dilating the original inner mask with extra iterations
        mask_dilated = cv2.dilate(mask_inner, kernel_small, iterations=4)  # Increased iterations from 3 to 4
        rim_mask = cv2.subtract(mask_dilated, mask_inner)
        if cv2.countNonZero(rim_mask) > 0:
            mean_rim = cv2.mean(lab_image, mask=rim_mask)[:3]
            diff = np.linalg.norm(np.array(mean_inside) - np.array(mean_rim))
        else:
            diff = 0
        
        # New: Identify inner region gaps using grayscale thresholding logic
        inner_gray = cv2.bitwise_and(gray_for_gap, gray_for_gap, mask=mask_inner)
        # Use a threshold similar to test4.py (50) with binary inversion to target darker pixels
        _, gap_thresh = cv2.threshold(inner_gray, 50, 255, cv2.THRESH_BINARY_INV)
        dark_pixels = cv2.countNonZero(gap_thresh)
        total_inner = cv2.countNonZero(mask_inner)
        gap_ratio = dark_pixels / total_inner if total_inner > 0 else 0
        
        # Combine both checks: either significant LAB difference or a gap ratio > 30%
        if diff > threshold or gap_ratio > 0.3:
            refined_inner_contours.append(cnt)
                
    # Draw only refined inner boundaries representing air voids in red
    for cnt in refined_inner_contours:
        area = cv2.contourArea(cnt)
        if area > 400 and area < 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Air Void", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
        
    # For demonstration, save processed image and return its path
    output_path = os.path.join('static', f"processed_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, image)
    return output_path


def load_defect_model():
    global loaded_model
    if loaded_model is None:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        num_classes = checkpoint["roi_heads.box_predictor.cls_score.weight"].shape[0]
        temp_model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
        temp_model.load_state_dict(checkpoint, strict=True)
        temp_model.eval()
        loaded_model = temp_model
    return loaded_model

def transform_image(image_path):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def predict_defects(model, image_tensor):
    with torch.no_grad():
        predictions = model([image_tensor])
    return predictions

def draw_predictions(image_path, predictions, labels_map, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    boxes = predictions[0]["boxes"]
    scores = predictions[0]["scores"]
    labels = predictions[0]["labels"]

    for i, box in enumerate(boxes):
        score = scores[i].cpu().item()
        if score > threshold:
            coords = box.cpu().numpy().astype(int)
            cv2.rectangle(image_np, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 255), 2)
            defect_label_id = int(labels[i].cpu().item())
            defect_label = labels_map.get(defect_label_id, "Other Defect")
            cv2.putText(image_np, f"{defect_label}", (coords[0], coords[3] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    output_path = os.path.join('static', f"{uuid.uuid4()}.png")
    cv2.imwrite(output_path, image_np)
    return output_path

# New: Cleanup function to delete files in the static folder older than max_age_seconds (e.g., 600 seconds)
def clean_static_folder(max_age_seconds=600):
    static_folder = 'static'
    now = time.time()
    for filename in os.listdir(static_folder):
        file_path = os.path.join(static_folder, filename)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path} due to age: {file_age} seconds")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {str(e)}")

@app.route("/detect", methods=["POST"])
@jwt_required()
def detect_defects():
    print("Debug: /detect route hit")
    clean_static_folder()  # new: clean old files at the start
    try:
        # Change 'file' to 'file1' for consistency with your frontend
        if 'file1' not in request.files:
            print("Debug: file1 not found in request")
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['file1']
        print(f"Debug: Received file: {file.filename}")
        if not file:
            return jsonify({"error": "Empty file"}), 400

        temp_image_path = os.path.join('static', f"{uuid.uuid4()}.png")
        file.save(temp_image_path)
        
        # Load or get the model
        model = load_defect_model()

        # Transform input
        image_tensor = transform_image(temp_image_path)

        # Predict
        print("Debug: Model loaded, performing prediction")
        predictions = predict_defects(model, image_tensor)

        # Draw predictions
        labels_map = {
            0: "arivoid", 1: "Conductor Show", 2: "Overlapping wings", 3: "Terminal sheet cracked",
            4: "Wing folding back", 5: "Wings Dissymmetry", 6: "Wings are not in contact",
            7: "Wings touching terminal floor or wall", 8: "Wrong cross section uploaded",
            9: "arivoid", 10: "burr", 11: "ok crimp"
        }
        final_image = draw_predictions(temp_image_path, predictions, labels_map)
        annotated_image_path = process_image(final_image,threshold=20)
        print(f"Debug: Annotated image saved at {annotated_image_path}")

        # Create a download link
        download_url = f"http://10.245.146.250:5005/download_annotated/{os.path.basename(annotated_image_path)}"

        # Optionally return predictions as JSON for front-end
        final_data = {
            "annotated_image_url": download_url,
            "predictions": [
                {
                    "box": predictions[0]["boxes"][i].cpu().numpy().tolist(),
                    "score": float(predictions[0]["scores"][i].cpu().numpy()),
                    "label": labels_map.get(int(predictions[0]["labels"][i].cpu().numpy()), "Other Defect")
                }
                for i in range(len(predictions[0]["boxes"]))
            ]
        }
        return jsonify(final_data), 200
    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download_annotated/<filename>', methods=['GET'])
def download_annotated(filename):
    print(f"Debug: /download_annotated route hit with filename={filename}")
    file_path = safe_join('static', filename)
    if not os.path.exists(file_path):
        abort(404)
    print(f"Debug: Sending annotated file {file_path}")
    response = send_file(file_path, as_attachment=True, mimetype='image/png')
    @response.call_on_close
    def remove_file():
        try:
            os.unlink(file_path)
            print(f"Deleted annotated file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {str(e)}")
    return response
 
if __name__ == "__main__":
    app.run(port="5005", host="0.0.0.0", debug=True)
