from flask import Flask, request, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import pandas as pd
from webscrapping import perform_clustering  # Import clustering function from webscrapping.py
from flask_cors import CORS
import uuid
# Initialize Flask application
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///job_portal.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your_secret_key'
CORS(app, resources={r"/*": {"origins": "*"}}) 


# Initialize SQLAlchemy and JWTManager
db = SQLAlchemy(app)
jwt = JWTManager(app)

user_applications = {}

@app.route('/apply-job', methods=['POST'])
def apply_job():
    data = request.json
    user_id = data.get('user_id')
    job = data.get('job')

    if not user_id or not job:
        return jsonify({'error': 'User ID and Job data are required'}), 400

    if user_id not in user_applications:
        user_applications[user_id] = []

    user_applications[user_id].append(job)
    return jsonify({'message': 'Job applied successfully', 'applications': user_applications[user_id]}), 200

@app.route('/user-applications/<user_id>', methods=['GET'])
def get_user_applications(user_id):
    if user_id not in user_applications:
        return jsonify({'message': 'No applications found for this user'}), 404

    return jsonify({'applications': user_applications[user_id]}), 200

# Model for the User table
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# Create tables and add initial data if necessary
with app.app_context():
    db.create_all()

# Route for user registration
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    # Validate required fields
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')

    if not email or not username or not password:
        return jsonify({"message": "Email, username, and password are required."}), 400

    # Check if the user already exists
    if User.query.filter_by(username=username).first():
        return jsonify({"message": "Username already exists."}), 400

    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully."}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    if not user or not check_password_hash(user.password, password):
        return jsonify({"message": "Invalid username or password"}), 401

    access_token = create_access_token(identity={"id": user.id, "username": user.username})
    return jsonify({
        "message": "Login successful",
        "access_token": access_token,
        "user_id": user.id  # Pastikan user_id dikirim dalam response
    }), 200



# Route for adding a job (authenticated users only)
@app.route('/add_job', methods=['POST'])
@jwt_required()
def add_job():
    data = request.get_json()
    new_job = {
        'Title': data.get('title'),
        'Company': data.get('company_name'),
        'Location': data.get('location'),
        'Work Type': data.get('work_type'),
        'Classification': data.get('classification')
    }

    global df
    df = pd.concat([df, pd.DataFrame([new_job])], ignore_index=True)
    return jsonify({"message": "Job added successfully."}), 201

# Route for searching jobs
@app.route('/search', methods=['GET'])
@jwt_required()
def search_jobs():
    title = request.args.get('title', '').lower()
    work_type = request.args.get('work_type', '').lower()
    classification = request.args.get('classification', '').lower()

    if df.empty:
        return jsonify({"message": "Job data not available."}), 500

    filtered_df = df[
        (df['Title'].str.lower().str.contains(title)) &
        (df['Work Type'].str.lower().str.contains(work_type)) &
        (df['Classification'].str.lower().str.contains(classification))
    ]

    if not filtered_df.empty:
        result = filtered_df[['Title', 'Company Name', 'Location', 'Work Type', 'Classification']].to_dict(orient='records')
        return jsonify(result)
    else:
        return jsonify({"message": "No jobs found matching the criteria."})

# Route to start clustering process
@app.route('/start-clustering', methods=['GET'])
def start_clustering():
    try:
        raw_df = pd.read_csv('job_data.csv')  # Ensure this file exists before clustering
        perform_clustering(raw_df)
        return jsonify({"message": "Clustering completed successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to download clustering result as CSV
@app.route('/download/csv', methods=['GET'])
def download_csv():
    try:
        return send_file('job_data_with_clusters.csv',
                         as_attachment=True,
                         mimetype='text/csv',
                         download_name='job_data_with_clusters.csv')
    except FileNotFoundError:
        return jsonify({"error": "File not found. Please run clustering first."}), 404

# Route to download clustering result as Excel
@app.route('/download/excel', methods=['GET'])
def download_excel():
    try:
        return send_file('job_data_with_clusters.xlsx',
                         as_attachment=True,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         download_name='job_data_with_clusters.xlsx')
    except FileNotFoundError:
        return jsonify({"error": "File not found. Please run clustering first."}), 404

# Route to view clustering data as JSON
@app.route('/view-data', methods=['GET'])
def view_data():
    try:
        df = pd.read_csv('job_data_with_clusters.csv')
        return jsonify(df.to_dict(orient='records')), 200
    except FileNotFoundError:
        return jsonify({"error": "File not found. Please run clustering first."}), 404

# Route to get all jobs
@app.route('/jobs', methods=['GET'])
def get_jobs():
    df = pd.read_csv('job_data_with_clusters.csv')
    df = df.fillna('')  # Replace NaN with empty string
    jobs = df.to_dict(orient='records')
    return jsonify(jobs)

# Route to filter jobs by cluster
@app.route('/jobs/cluster/<cluster_name>', methods=['GET'])
def get_jobs_by_cluster(cluster_name):
    df = pd.read_csv('job_data_with_clusters.csv')
    df = df.fillna('')  # Replace NaN with empty string

    # Ensure 'Cluster' column is a string
    df['Cluster'] = df['Cluster'].astype(str)

    # Filter jobs by cluster
    filtered_df = df[df['Cluster'] == cluster_name]

    if filtered_df.empty:
        return jsonify({"message": f"No jobs found for cluster '{cluster_name}'"}), 404

    jobs = filtered_df.to_dict(orient='records')
    return jsonify(jobs)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)