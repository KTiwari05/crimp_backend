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
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
 
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

@app.route("/detect", methods=["POST"])
@jwt_required()
def detect_defects():
    print("Debug: /detect route hit")
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
        annotated_image_path = draw_predictions(temp_image_path, predictions, labels_map)
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
    return send_file(file_path, as_attachment=True, mimetype='image/png')
 
if __name__ == "__main__":
    app.run(port="5005", host="0.0.0.0", debug=True)
