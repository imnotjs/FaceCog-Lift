from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from deepface import DeepFace
import os
import numpy as np
from PIL import Image, ImageFilter

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session management

# Path to folder containing authorized faces
AUTHORIZED_FACES_DIR = "authorized_faces"

# Precompute embeddings for authorized faces
AUTHORIZED_EMBEDDINGS = {}
for filename in os.listdir(AUTHORIZED_FACES_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # Accept image formats
        face_path = os.path.join(AUTHORIZED_FACES_DIR, filename)
        try:
            # Compute embedding for each face image
            embedding = DeepFace.represent(img_path=face_path, model_name="VGG-Face")[0]["embedding"]
            person_name = os.path.splitext(filename)[0]  # Use the filename (without extension) as the name
            AUTHORIZED_EMBEDDINGS[person_name] = np.array(embedding)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def preprocess_image(image_path):
    """
    Preprocess the image by resizing, normalizing, applying minimal filtering, and mirroring it.
    """
    image = Image.open(image_path)
    
    # If the image has an alpha channel (RGBA), convert it to RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Mirror the image (flip horizontally)
    image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Apply unsharp mask to reduce blurriness and sharpen image
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # Resize the image for consistency (160x160 is a typical input size for face recognition)
    image = image.resize((160, 160))

    # Save the preprocessed image temporarily
    preprocessed_path = "preprocessed_image.jpg"
    image.save(preprocessed_path, "JPEG")
    return preprocessed_path

# Initial lift level
current_level = 1

@app.route("/")
def home():
    session.clear()  # Clear session data on page load
    return render_template("index.html")

@app.route("/face-login")
def face_login():
    return render_template("face-login.html")

@app.route("/authenticate", methods=["POST"])
def authenticate():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image uploaded."}), 400

    image = request.files["image"]
    image_path = "uploaded_image.jpg"
    image.save(image_path)

    # Preprocess the image
    preprocessed_image_path = preprocess_image(image_path)

    try:
        # Compute embedding for the preprocessed image
        uploaded_embedding = np.array(DeepFace.represent(img_path=preprocessed_image_path, model_name="VGG-Face")[0]["embedding"])

        # Compare embeddings with authorized faces
        for name, authorized_embedding in AUTHORIZED_EMBEDDINGS.items():
            similarity = np.dot(uploaded_embedding, authorized_embedding) / (
                np.linalg.norm(uploaded_embedding) * np.linalg.norm(authorized_embedding)
            )

            print(f"Comparing with {name}: similarity = {similarity}")

            if similarity > 0.40:  # Threshold for similarity
                session['authenticated'] = True
                session['user_name'] = name
                return jsonify({"status": "success", "redirect_to": url_for('profile')})

        return jsonify({"status": "error", "message": "Face not recognized. Please try again."})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error processing image: {e}"}), 500

@app.route("/profile")
def profile():
    if not session.get('authenticated', False):
        return redirect(url_for('home'))  # Redirect to homepage if not authenticated
    user_name = session.get('user_name', 'Unknown User')
    return render_template("profile.html", user_name=user_name, authorized_levels=[1, 2])

@app.route("/level_control")
def level_control():
    if not session.get('authenticated', False):
        return redirect(url_for('home'))  # Redirect to homepage if not authenticated
    return render_template("level_control.html", current_level=current_level)

@app.route("/get_level", methods=["GET"])
def get_level():
    global current_level
    return jsonify({"current_level": current_level})

@app.route("/update_level", methods=["POST"])
def update_level():
    """
    Public endpoint for ESP32 to update the current lift level.
    """
    global current_level
    level = request.form.get("level")
    if level not in ["1", "2", "Moving"]:
        return jsonify({"status": "error", "message": "Invalid level value."}), 400
    
    current_level = level
    print(f"Level updated to: {current_level}")
    return jsonify({"status": "success", "current_level": current_level})

@app.route("/change_level", methods=["POST"])
def change_level():
    if not session.get('authenticated', False):
        return jsonify({"status": "error", "message": "Unauthorized access. Please authenticate first."}), 403

    requested_level = request.form.get('level')
    try:
        level = int(requested_level)
        if level in [1, 2]:  # Allowed levels
            global current_level
            current_level = level
            return jsonify({"status": "success", "new_level": current_level})
        else:
            return jsonify({"status": "error", "message": "Invalid level."}), 400
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid input. Level must be a number."}), 400

@app.route("/face-logout")
def face_logout():
    return render_template("face-logout.html")

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
