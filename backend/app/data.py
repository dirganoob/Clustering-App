from flask import Flask, request, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import pandas as pd
from webscrapping import perform_clustering  # Import clustering function from webscrapping.py
from flask_cors import CORS

 # Izinkan permintaan dari frontend


# Initialize Flask application
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///job_portal.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your_secret_key'
CORS(app) 
# Initialize SQLAlchemy and JWTManager
db = SQLAlchemy(app)
jwt = JWTManager(app)

# Model for the User table
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# Create tables and add admin user if not exists
with app.app_context():
    db.create_all()
    # Check if admin exists, if not, create admin
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        hashed_password = generate_password_hash('jobstreet')  # Password for admin
        admin_user = User(username='admin', password=hashed_password)
        db.session.add(admin_user)
        db.session.commit()
        print("Admin user created with username 'admin' and password 'jobstreet'.")

# Route for user registration (no admin creation here)
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    
    # Validate required fields
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')

    if not email or not username or not password:
        return jsonify({"message": "Email, username, and password are required."}), 400

    # Prevent creating 'admin' user
    if username.lower() == 'admin':
        return jsonify({"message": "Username 'admin' is reserved."}), 400

    # Check if the user already exists
    if User.query.filter_by(username=username).first():
        return jsonify({"message": "Username already exists."}), 400

    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully."}), 201

# Route for user login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        is_admin = (user.username == 'admin')  # Check if the user is admin
        access_token = create_access_token(identity={"username": user.username, "is_admin": is_admin})
        return jsonify({"access_token": access_token, "is_admin": is_admin}), 200

    return jsonify({"message": "Invalid username or password."}), 401

# Route for admin to view all users
@app.route('/admin/view_users', methods=['GET'])
@jwt_required()
def view_users():
    current_user = get_jwt_identity()
    if current_user['username'] != 'admin':
        return jsonify({"message": "Admin access required."}), 403

    users = User.query.all()
    users_data = [{"id": user.id, "username": user.username} for user in users if user.username != 'admin']
    return jsonify(users_data), 200

# Admin route for adding a job
@app.route('/admin/add_job', methods=['POST'])
@jwt_required()
def add_job():
    current_user = get_jwt_identity()

    if current_user['username'] != 'admin':
        return jsonify({"message": "Admin access required."}), 403

    data = request.get_json()
    new_job = {
        'Title': data.get('title'),
        'Company Name': data.get('company_name'),
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

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
   
