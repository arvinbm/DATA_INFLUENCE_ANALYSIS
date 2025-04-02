from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os

from loo_shapley import get_loo_shapely_results

app = Flask(__name__)

# Define and create a folder to hold the uploaded csv file
UPLOAD_FILES_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FILES_FOLDER, exist_ok=True)
app.config['UPLOADED_FILES_FOLDER'] = UPLOAD_FILES_FOLDER

# Serve the index.html
@app.route('/')
def serve_index(): 
    return render_template('index.html')

# Handle file upload and influence analysis
@app.route('/upload', methods=['POST'])
def handle_uploaded_file():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({"ERROR" : "No File Uploaded"}), 400 # User Error
    
    # Check if a file was selected by the user
    uploaded_filename = request.files["file"]
    if uploaded_filename == '':
        return jsonify({"ERROR": "No File Selected"}), 400 
    
    # Save the uploaded file in the specified folder
    # Remove any unwanted characters in the filename which could have been added by the browser
    uploaded_filename_secure = secure_filename(uploaded_filename.filename)
    filepath = os.path.join (app.config["UPLOADED_FILES_FOLDER"], uploaded_filename_secure)
    uploaded_filename.save(filepath)

    try:
        # Obtain the most influential data points 
        result = get_loo_shapely_results(filepath)
        return jsonify(result)
    
    except Exception as error:
        return jsonify({"ERROR" : str(error)}), 500 # Server Error 

if __name__ == "__main__":
    app.run(debug=True)