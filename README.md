# Face Recognition IOT-Based Lift (for Final Project Purpose)

This Flask-based application implements a facial recognition authentication system using **DeepFace (VGG-Face model)** to control access to a two-level IoT lift. Authenticated user sessions are managed via Flask's session object. Upon successful facial recognition, users gain access to a control panel that communicates with an ESP32 microcontroller to update or retrieve the current lift level. The system uses **PIL** for image preprocessing, including resizing, mirroring, and filtering, and NumPy for embedding comparison.

## Module and Technologies

Here, we list things that we used upon this projects.

### 1. Flask
Flask is a Python micro web framework used to create and route HTTP endpoints, manage sessions, and render dynamic HTML using Jinja2. It powers the entire server-side logic of the system. User interaction, session flow, and data exchange with hardware are all handled via Flask routes.

### 2. DeepFace (VGG-Face)
DeepFace is a deep learning library used here for real-time facial recognition, specifically leveraging the VGG-Face model to compute 2622-dimensional embeddings. These embeddings represent facial features and are compared using cosine similarity. It enables secure, contactless login by matching uploaded images to pre-authorized profiles.

### 3. PIL (Pillow)
Pillow is used for image preprocessing, which includes resizing, unsharp masking, and horizontal flipping to standardize input for better facial recognition accuracy. This step ensures image uniformity before embedding extraction. It helps mitigate issues caused by lighting, resolution, or orientation.

### 4. ESP32
ESP32 acts as the IoT controller that physically drives the lift system based on commands received from the Flask server. It also reports the current floor level by posting updates to the backend. This bidirectional communication allows for real-time lift monitoring and control.

### 5. NGROK
Ngrok creates a secure tunnel from a public URL to your locally hosted Flask server, allowing the ESP32 and users to access it from outside the local network. It’s essential for development and testing in scenarios without a public IP. The tunnel is dynamically assigned and supports HTTPS. We use NGROK to expose our local Flask server over HTTPS, as **camera access requires a secure (HTTPS) context to function in modern browsers**.
