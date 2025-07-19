import logging
import os
import cv2
import torch
import random
import shutil
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, Response, flash, session, current_app as app
from werkzeug.utils import secure_filename, safe_join
from ultralytics import YOLO
import glob

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for flashing messages

# Set upload and frame folders
UPLOAD_FOLDER = 'static/uploads'
SAVED_FRAMES_FOLDER = 'static/saved_frames/'
ANNOTATIONS_FOLDER = 'static/annotations/'
ORIGINAL_FRAMES_FOLDER = 'static/original_frames/'
FINETUNED_MODELS_FOLDER = "finetunedmodels"

# Ensure necessary directories exist
os.makedirs(SAVED_FRAMES_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)
os.makedirs(ORIGINAL_FRAMES_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FINETUNED_MODELS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAVED_FRAMES_FOLDER'] = SAVED_FRAMES_FOLDER
app.config['ANNOTATIONS_FOLDER'] = ANNOTATIONS_FOLDER
app.config['ORIGINAL_FRAMES_FOLDER'] = ORIGINAL_FRAMES_FOLDER

# Dataset folder and subfolders
DATASET_FOLDER = 'static/prepared_dataset/'
TRAIN_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, 'images/train')
VAL_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, 'images/val')
TRAIN_LABELS_FOLDER = os.path.join(DATASET_FOLDER, 'labels/train')
VAL_LABELS_FOLDER = os.path.join(DATASET_FOLDER, 'labels/val')

# Ensure dataset folders exist
os.makedirs(TRAIN_IMAGES_FOLDER, exist_ok=True)
os.makedirs(VAL_IMAGES_FOLDER, exist_ok=True)
os.makedirs(TRAIN_LABELS_FOLDER, exist_ok=True)
os.makedirs(VAL_LABELS_FOLDER, exist_ok=True)

# Google Cloud configuration (set your credentials file path)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/saisannidh535354/mount/app/flaskvideo/weapon-detection-project-22adcac55981.json"

# Path to training script
TRAINING_SCRIPT_PATH = '/home/saisannidh535354/mount/app/flaskvideo/training_script.py'

# Default model path
DEFAULT_MODEL_PATH = 'weights/best.pt'
model = None  # YOLOv8 model placeholder

# Load the model
def initialize_model():
    """Initialize YOLOv8 model before handling requests."""
    load_model()

def load_model(selected_model_path=None):
    """Load the YOLO model."""
    global model
    selected_model_path = selected_model_path or session.get('selected_model_path', DEFAULT_MODEL_PATH)
    
    try:
        model = YOLO(selected_model_path)  # Load the YOLOv8 model
        logging.info(f"Successfully loaded model: {os.path.basename(selected_model_path)}", "success")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")

@app.route('/')
def index():
    """Home page to upload a video or use the live camera feed."""
    return render_template('index.html')

@app.route('/select_model', methods=['GET', 'POST'])
def select_model():
    """Allow user to select a YOLO model for detection."""
    models_folder = 'finetunedmodels'
    base_model = 'weights/best.pt'
    
    # List available models (base + finetuned models)
    models = [base_model] + sorted(glob.glob(f'{models_folder}/*.pt'), key=os.path.getmtime, reverse=True)
    
    if request.method == 'POST':
        selected_model = request.form.get('model')
        if selected_model:
            if selected_model == base_model:
                session['selected_model_path'] = selected_model
                flash(f"Selected model: {os.path.basename(selected_model)}", "success")
            elif selected_model.startswith(models_folder):
                weights_path = os.path.join(selected_model, 'weights/best.pt')
                if os.path.isfile(weights_path):
                    session['selected_model_path'] = weights_path
                    flash(f"Selected fine-tuned model: {weights_path}", "success")
                else:
                    flash(f"Error: '{weights_path}' is not a valid model file.", "error")
            else:
                flash(f"Error: The selected path '{selected_model}' is not valid.", "error")
        else:
            flash("No model selected.", "error")
        return redirect(url_for('select_model'))
    
    return render_template('select_model.html', models=models)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handles video file upload and redirects to detection."""
    if 'video' not in request.files:
        flash("No video file uploaded.")
        return redirect('/')
    
    file = request.files['video']
    if file.filename == '':
        flash("No file selected.")
        return redirect('/')
    
    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)
        logging.info(f"Uploaded video saved at {video_path}")
        return redirect(url_for('detect_video_stream', video_path=filename))

@app.route('/detect_video/<path:video_path>', endpoint='detect_video_stream')
def detect_video(video_path):
    """Streams the processed video with detections."""
    decoded_video_path = safe_join(UPLOAD_FOLDER, video_path)
    selected_model_path = session.get('selected_model_path', DEFAULT_MODEL_PATH)
    
    if not os.path.exists(decoded_video_path):
        flash(f"Video not found: {decoded_video_path}")
        return redirect('/')
    
    logging.info(f"Processing video at {decoded_video_path}")
    
    return Response(
        gen_frames(cv2.VideoCapture(decoded_video_path), selected_model_path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/view_saved_frames')
def view_saved_frames():
    """Displays a list of saved frames."""
    saved_frames = sorted([f for f in os.listdir(SAVED_FRAMES_FOLDER) if f.endswith('.jpg')])
    if not saved_frames:
        flash("No frames saved. Please process a video first.")
        return redirect('/')
    
    return render_template('view_saved_frames.html', saved_frames=saved_frames)

@app.route('/show_frame/<int:frame_index>', methods=['GET', 'POST'])
def show_frame(frame_index):
    """Displays a specific frame with options to delete or continue."""
    saved_frames = sorted([f for f in os.listdir(SAVED_FRAMES_FOLDER) if f.endswith('.jpg')])
    
    if frame_index >= len(saved_frames):
        flash("No more frames to display.")
        return redirect('/view_saved_frames')
    
    frame_filename = saved_frames[frame_index]
    annotation_filename = os.path.join(ANNOTATIONS_FOLDER, frame_filename.replace('.jpg', '.txt'))
    original_frame_filename = os.path.join(ORIGINAL_FRAMES_FOLDER, frame_filename)
    
    if request.method == 'POST':
        action = request.form['action']
        if action == 'delete':
            os.remove(os.path.join(SAVED_FRAMES_FOLDER, frame_filename))
            os.remove(annotation_filename) if os.path.exists(annotation_filename) else None
            os.remove(original_frame_filename) if os.path.exists(original_frame_filename) else None
            flash(f"Deleted frame: {frame_filename}")
            return redirect(url_for('show_frame', frame_index=frame_index))
        elif action == 'continue':
            return redirect(url_for('show_frame', frame_index=frame_index + 1))
    
    return render_template('show_frame.html', frame_path=frame_filename, frame_index=frame_index)

def gen_frames(camera, selected_model_path=None):
    """Generates frames from video or live camera feed with detections."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Load model if not loaded
        if model is None:
            load_model(selected_model_path)
        
        # Perform detection
        detected_frame, annotations = detect_objects(frame, selected_model_path)
        
        if annotations:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            
            # Save the original frame
            original_frame_path = os.path.join(app.config['ORIGINAL_FRAMES_FOLDER'], f"frame_{timestamp}.jpg")
            cv2.imwrite(original_frame_path, frame)
            
            # Save frame with bounding boxes
            frame_path = os.path.join(app.config['SAVED_FRAMES_FOLDER'], f"frame_{timestamp}.jpg")
            cv2.imwrite(frame_path, detected_frame)
            
            # Save YOLO annotations
            annotation_path = os.path.join(app.config['ANNOTATIONS_FOLDER'], f"frame_{timestamp}.txt")
            with open(annotation_path, 'w') as f:
                f.write("\n".join(annotations))
        
        # Yield the frame for streaming
        ret, buffer = cv2.imencode('.jpg', detected_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def detect_objects(frame, selected_model_path):
    """Detects objects using YOLOv8 and returns annotated frame and YOLO annotations."""
    global model
    if model is None:
        load_model(selected_model_path)
    
    results = model(frame)
    result = results[0] if isinstance(results, list) else results
    
    annotations = []
    for box in result.boxes.data:
        x1, y1, x2, y2, confidence, class_id = box.tolist()
        if confidence > 0.3:
            x_center = (x1 + x2) / 2 / frame.shape[1]
            y_center = (y1 + y2) / 2 / frame.shape[0]
            width = (x2 - x1) / frame.shape[1]
            height = (y2 - y1) / frame.shape[0]
            annotations.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    detected_frame = result.plot()  # Plot bounding boxes on the frame
    return detected_frame, annotations

@app.route('/live_feed')
def live_feed():
    """Streams the live camera feed with detections."""
    camera = cv2.VideoCapture(1)  # Use 0 for default camera or change the index for others
    
    if not camera.isOpened():
        flash("Error: Unable to access camera.")
        return redirect('/')
    
    return Response(
        gen_frames(camera, session.get('selected_model_path', DEFAULT_MODEL_PATH)),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
@app.route('/prepare_dataset', methods=['POST'])
def prepare_dataset():
    """Prepares the dataset with an 80/20 split and generates a YAML file."""
    # Gather images and annotations
    images = [f for f in os.listdir(ORIGINAL_FRAMES_FOLDER) if f.endswith('.jpg')]
    annotations = [f.replace('.jpg', '.txt') for f in images]
    paired_files = list(zip(images, annotations))

    if not paired_files:
        flash("No images or annotations found to prepare the dataset.")
        return redirect('/')

    # Shuffle and split into train and val
    random.shuffle(paired_files)
    split_index = int(0.8 * len(paired_files))
    train_pairs = paired_files[:split_index]
    val_pairs = paired_files[split_index:]

    # Move files to their respective folders
    def move_files(pairs, img_dest, lbl_dest):
        for img, lbl in pairs:
            src_img_path = os.path.join(ORIGINAL_FRAMES_FOLDER, img)
            src_lbl_path = os.path.join(ANNOTATIONS_FOLDER, lbl)
            dest_img_path = os.path.join(img_dest, img)
            dest_lbl_path = os.path.join(lbl_dest, lbl)

            if os.path.exists(src_img_path) and os.path.exists(src_lbl_path):
                shutil.copy(src_img_path, dest_img_path)
                shutil.copy(src_lbl_path, dest_lbl_path)
            else:
                logging.warning(f"Missing file for pair: {img}, {lbl}")

    move_files(train_pairs, TRAIN_IMAGES_FOLDER, TRAIN_LABELS_FOLDER)
    move_files(val_pairs, VAL_IMAGES_FOLDER, VAL_LABELS_FOLDER)

    # Generate YAML file
    yaml_content = f"""
train: {os.path.abspath(TRAIN_IMAGES_FOLDER)}
val: {os.path.abspath(VAL_IMAGES_FOLDER)}
nc: 1
names: ['weapon']
    """
    yaml_path = os.path.join(DATASET_FOLDER, 'dataset.yaml')
    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml_content)

    flash("Dataset prepared successfully!")
    return redirect('/')

@app.route('/start_fine_tuning', methods=['POST'])
def start_fine_tuning():
    """Handle the fine-tuning process when button is clicked."""
    # Get the selected model path from the session or fallback to default
    selected_model_path = session.get('selected_model_path', 'weights/best.pt')

    # Ensure selected model exists
    if not os.path.isfile(selected_model_path):
        flash(f"Selected model '{selected_model_path}' not found. Please select a valid model.", 'error')
        return redirect(url_for('index'))

    dataset_yaml = 'static/prepared_dataset/dataset.yaml'
    epochs = 10
    project_folder = FINETUNED_MODELS_FOLDER
    model_name = f'fine_tuned_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'

    # Build the YOLO fine-tuning command
    command = [
        'yolo', 'train',
        f'model={selected_model_path}',
        f'data={dataset_yaml}',
        f'epochs={epochs}',
        f'project={project_folder}',
        f'name={model_name}'
    ]

    try:
        # Run the YOLO fine-tuning command
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            error_message = f"Fine-tuning failed: {result.stderr}"
            flash(error_message, 'error')
            return redirect(url_for('index'))

        success_message = f"Fine-tuned model saved: {model_name}"
        flash(success_message, 'success')
        return redirect(url_for('index'))

    except Exception as e:
        flash(f"Error during fine-tuning: {str(e)}", 'error')
        return redirect(url_for('index'))
        
@app.route('/prepare_dataset', methods=['POST'])
def prepare_dataset():
    """Prepares the dataset with an 80/20 split and generates a YAML file."""
    # Gather images and annotations
    images = [f for f in os.listdir(ORIGINAL_FRAMES_FOLDER) if f.endswith('.jpg')]
    annotations = [f.replace('.jpg', '.txt') for f in images]
    paired_files = list(zip(images, annotations))

    if not paired_files:
        flash("No images or annotations found to prepare the dataset.")
        return redirect('/')

    # Shuffle and split into train and val
    random.shuffle(paired_files)
    split_index = int(0.8 * len(paired_files))
    train_pairs = paired_files[:split_index]
    val_pairs = paired_files[split_index:]

    # Move files to their respective folders
    def move_files(pairs, img_dest, lbl_dest):
        for img, lbl in pairs:
            src_img_path = os.path.join(ORIGINAL_FRAMES_FOLDER, img)
            src_lbl_path = os.path.join(ANNOTATIONS_FOLDER, lbl)
            dest_img_path = os.path.join(img_dest, img)
            dest_lbl_path = os.path.join(lbl_dest, lbl)

            if os.path.exists(src_img_path) and os.path.exists(src_lbl_path):
                shutil.copy(src_img_path, dest_img_path)
                shutil.copy(src_lbl_path, dest_lbl_path)
            else:
                logging.warning(f"Missing file for pair: {img}, {lbl}")

    move_files(train_pairs, TRAIN_IMAGES_FOLDER, TRAIN_LABELS_FOLDER)
    move_files(val_pairs, VAL_IMAGES_FOLDER, VAL_LABELS_FOLDER)

    # Generate YAML file
    yaml_content = f"""
train: {os.path.abspath(TRAIN_IMAGES_FOLDER)}
val: {os.path.abspath(VAL_IMAGES_FOLDER)}
nc: 1
names: ['weapon']
    """
    yaml_path = os.path.join(DATASET_FOLDER, 'dataset.yaml')
    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml_content)

    flash("Dataset prepared successfully!")
    return redirect('/')

@app.route('/start_fine_tuning', methods=['POST'])
def start_fine_tuning():
    """Handle the fine-tuning process when button is clicked."""
    # Get the selected model path from the session or fallback to default
    selected_model_path = session.get('selected_model_path', 'weights/best.pt')

    # Ensure selected model exists
    if not os.path.isfile(selected_model_path):
        flash(f"Selected model '{selected_model_path}' not found. Please select a valid model.", 'error')
        return redirect(url_for('index'))

    dataset_yaml = 'static/prepared_dataset/dataset.yaml'
    epochs = 10
    project_folder = FINETUNED_MODELS_FOLDER
    model_name = f'fine_tuned_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'

    # Build the YOLO fine-tuning command
    command = [
        'yolo', 'train',
        f'model={selected_model_path}',
        f'data={dataset_yaml}',
        f'epochs={epochs}',
        f'project={project_folder}',
        f'name={model_name}'
    ]

    try:
        # Run the YOLO fine-tuning command
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            error_message = f"Fine-tuning failed: {result.stderr}"
            flash(error_message, 'error')
            return redirect(url_for('index'))

        success_message = f"Fine-tuned model saved: {model_name}"
        flash(success_message, 'success')
        return redirect(url_for('index'))

    except Exception as e:
        flash(f"Error during fine-tuning: {str(e)}", 'error')
        return redirect(url_for('index'))
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=5000, debug=True)
