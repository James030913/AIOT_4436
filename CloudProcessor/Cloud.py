import paho.mqtt.client as mqtt
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import tempfile
import json
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import joblib
import sqlite3
import os
import requests

# --- Configuration ---
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC_SUB = "health/user_001/vitals"  # Topic to subscribe to
FLASK_PORT = 5000
DB_PATH = 'health_data.db'
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your-api-key-here')  # Set your API key in environment variables
# Feature list
ML_FEATURES = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 'oxygen_saturation']

risk_mapping = {'Low Risk': 0, 'High Risk': 1}
# Alert thresholds (simple rules)
ALERT_THRESHOLDS = {
    'heart_rate': (45, 120), 'systolic_bp': (85, 140), 'diastolic_bp': (55, 90),
    'temperature': (35.0, 38.5), 'oxygen_saturation': (92, 101),
}

# --- Database Setup Function ---
def setup_database():
    """Create SQLite database and tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table for vital signs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vital_signs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        heart_rate REAL,
        systolic_bp REAL,
        diastolic_bp REAL,
        temperature REAL,
        oxygen_saturation REAL,
        activity_level TEXT,
        risk_category TEXT,
        risk_probability REAL
    )
    ''')
    
    # Create table for alerts
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        alert_message TEXT NOT NULL
    )
    ''')
    
    # Create table for users
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL UNIQUE,
        username TEXT UNIQUE,
        password TEXT,
        name TEXT,
        height REAL,
        weight REAL,
        bmi REAL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

# --- Cloud Processor Class (modified to include SQLite storage) ---
class CloudProcessor:
    def __init__(self, user_id, ml_features):
        self.user_id = user_id
        self.historical_data = []
        self.ml_features = ml_features
        self.data_count = 0
        self.last_trend_check_time = None
        self._lock = threading.Lock()  # Lock for thread safety when accessing shared data
        
        # Ensure database exists
        if not os.path.exists(DB_PATH):
            setup_database()

    def store_vitals_in_db(self, vitals, risk_result=None):
        """Store vital signs in SQLite database"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Prepare data for insertion
            risk_category = risk_result.get('risk_category') if risk_result else None
            risk_probability = risk_result.get('probability') if risk_result else None
            
            cursor.execute('''
            INSERT INTO vital_signs 
            (user_id, timestamp, heart_rate, systolic_bp, diastolic_bp, temperature, oxygen_saturation, 
             activity_level, risk_category, risk_probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                vitals['user_id'],
                vitals['timestamp'],
                float(vitals.get('heart_rate', 0)),
                float(vitals.get('systolic_bp', 0)),
                float(vitals.get('diastolic_bp', 0)), 
                float(vitals.get('temperature', 0)),
                float(vitals.get('oxygen_saturation', 0)),
                vitals.get('activity_level', None),
                risk_category,
                risk_probability
            ))
            
            conn.commit()
            conn.close()
            print(f"Stored vital signs for {vitals['user_id']} in database")
            
        except Exception as e:
            print(f"Error storing vital signs in database: {e}")

    def store_alert_in_db(self, user_id, timestamp, alert_type, alert_message):
        """Store alert in SQLite database"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO alerts (user_id, timestamp, alert_type, alert_message)
            VALUES (?, ?, ?, ?)
            ''', (user_id, timestamp, alert_type, alert_message))
            
            conn.commit()
            conn.close()
            print(f"Alert stored in database for {user_id}")
            
        except Exception as e:
            print(f"Error storing alert in database: {e}")

    def process_live_data(self, vitals):
        timestamp_dt = datetime.fromisoformat(vitals['timestamp'])  # Convert string back to datetime
        vitals_processed = vitals.copy()  # Work with a copy
        vitals_processed['timestamp_dt'] = timestamp_dt  # Keep datetime object internally
        
        with self._lock:  # Acquire lock before modifying shared data
            self.historical_data.append(vitals_processed)
            self.data_count += 1

        # Convert incoming vitals to numeric for checks
        try:
            hr = float(vitals['heart_rate'])
            sys_bp = float(vitals['systolic_bp'])
            dia_bp = float(vitals['diastolic_bp'])
            temp = float(vitals['temperature'])
            spo2 = float(vitals['oxygen_saturation'])
        except (ValueError, TypeError) as e:
            print(f"[CloudProcessor] Warning: Could not parse vital sign numbers: {e}")
            return  # Cannot process this data point
        
        # Get risk prediction
        risk_result = predict_risk(hr, sys_bp, dia_bp, temp, spo2)
        print(f"Risk prediction: {risk_result}")
        
        # Store data in SQLite
        self.store_vitals_in_db(vitals, risk_result)
        
        # Simple Rule-based Check
        anomalies_found = []
        if not (ALERT_THRESHOLDS['heart_rate'][0] <= hr <= ALERT_THRESHOLDS['heart_rate'][1]):
            anomalies_found.append(f"Rule: Heart Rate abnormal: {hr} bpm")
        if not (ALERT_THRESHOLDS['systolic_bp'][0] <= sys_bp <= ALERT_THRESHOLDS['systolic_bp'][1]):
            anomalies_found.append(f"Rule: Systolic BP abnormal: {sys_bp} mmHg")
        if spo2 < ALERT_THRESHOLDS['oxygen_saturation'][0]:
            anomalies_found.append(f"Rule: Oxygen Saturation low: {spo2}%")
        
        # Send Alert if needed
        if anomalies_found:
            self.send_alert(anomalies_found, timestamp_dt)
        
        # Add risk information to alert if it's high
        if risk_result['risk_category'] in ['High', 'Very High']:
            risk_alert = f"Risk Assessment: {risk_result['risk_category']} risk detected (probability: {risk_result['probability']:.2f})"
            self.send_alert([risk_alert], timestamp_dt, is_risk_alert=True)

    def send_alert(self, anomalies, timestamp, is_risk_alert=False):
        alert_type = "Risk Alert" if is_risk_alert else "Rule-Based Alert"
        print("-" * 30)
        print(f"🚨 {alert_type} for {self.user_id} at {timestamp.strftime('%Y-%m-%d %H:%M:%S')} 🚨")
        for anomaly in anomalies:
            print(f"  - {anomaly}")
            # Store each alert in database
            self.store_alert_in_db(
                user_id=self.user_id,
                timestamp=timestamp.isoformat(),
                alert_type=alert_type,
                alert_message=anomaly
            )
        print("-" * 30)

    def perform_batch_analysis(self):
        print("\n[CloudProcessor] Performing batch analysis...")
        
        # Option 1: Use the data from memory
        with self._lock:  # Acquire lock to safely copy data
            if not self.historical_data:
                print("[CloudProcessor] No data available in memory for batch analysis.")
                # Fall back to database
                return self.perform_batch_analysis_from_db()
            # Create DataFrame from a copy of the historical data
            health_df = pd.DataFrame(list(self.historical_data))  # Ensure it's a copy
        
        if health_df.empty:
            print("[CloudProcessor] DataFrame is empty after copying.")
            return self.perform_batch_analysis_from_db()
        
        return self._analyze_dataframe(health_df)
    
    def perform_batch_analysis_from_db(self):
        """Get data from SQLite and perform analysis"""
        print("[CloudProcessor] Retrieving data from database...")
        try:
            conn = sqlite3.connect(DB_PATH)
            query = f"SELECT * FROM vital_signs WHERE user_id = '{self.user_id}' ORDER BY timestamp"
            health_df = pd.read_sql_query(query, conn)
            conn.close()
            
            if health_df.empty:
                print("[CloudProcessor] No data available in database for batch analysis.")
                return {}
            
            # Convert timestamp to datetime
            health_df['timestamp_dt'] = pd.to_datetime(health_df['timestamp'])
            return self._analyze_dataframe(health_df)
            
        except Exception as e:
            print(f"[CloudProcessor] Error retrieving data from database: {e}")
            return {}

    def _analyze_dataframe(self, health_df):
        """Analyze dataframe regardless of data source"""
        # Use the internal datetime object for indexing
        if 'timestamp_dt' in health_df.columns:
            health_df['timestamp'] = health_df['timestamp_dt']
        
        health_df = health_df.set_index('timestamp').sort_index()
        
        # Drop unnecessary columns
        drop_cols = ['timestamp_dt', 'id']
        health_df = health_df.drop(columns=[col for col in drop_cols if col in health_df.columns], errors='ignore')
        
        # Convert relevant columns to numeric *robustly*
        numeric_cols_potential = self.ml_features + ['activity_level']
        numeric_cols_present = [col for col in numeric_cols_potential if col in health_df.columns]
        for col in numeric_cols_present:
            health_df[col] = pd.to_numeric(health_df[col], errors='coerce')
        health_df = health_df.dropna(subset=numeric_cols_present)  # Drop rows where key numerics are NaN
        numeric_cols = health_df.select_dtypes(include=np.number).columns  # Get actual numeric cols
        
        if health_df.empty or numeric_cols.empty:
            print("[CloudProcessor] Not enough valid numeric data for batch analysis after cleaning.")
            return {}
        
        # --- Calculate Trends ---
        trends = {}
        try:
            daily_trends = health_df[numeric_cols].resample('D').agg(['mean', 'min', 'max']).round(2)
            trends['daily'] = daily_trends
            hourly_trends = health_df[numeric_cols].resample('H').agg(['mean', 'min', 'max']).round(2)
            trends['hourly'] = hourly_trends
            if 'heart_rate' in health_df.columns and len(health_df) > 10:
                # Use a decaying EMA, alpha = 2 / (span + 1)
                health_df['hr_ema'] = health_df['heart_rate'].ewm(alpha=0.1, adjust=False).mean()
                trends['hr_ema_latest'] = round(health_df['hr_ema'].iloc[-1], 2)
        except Exception as e:
            print(f"[CloudProcessor] Error calculating trends: {e}")
        
        # --- Calculate Baseline ---
        baseline = {}
        if len(health_df) >= 10:
            baseline = health_df[numeric_cols].mean().round(2).to_dict()
        
        # --- Simulate Pattern Recognition ---
        pattern_alerts = []
        try:
            if 'daily' in trends and not trends['daily'].empty and ('heart_rate', 'mean') in trends['daily'].columns:
                daily_mean_hr = trends['daily'][('heart_rate', 'mean')].dropna()
                if len(daily_mean_hr) >= 4:  # Need 4 points to compare last 3 to previous one
                    # Check if last 3 are strictly increasing
                    if all(np.diff(daily_mean_hr.iloc[-3:].values) > 0):
                        pattern_alerts.append("Pattern Alert: Mean daily heart rate has shown an increasing trend over the last 3 days.")
            
            if 'daily' in trends and not trends['daily'].empty and ('oxygen_saturation', 'mean') in trends['daily'].columns:
                daily_mean_spo2 = trends['daily'][('oxygen_saturation', 'mean')].dropna()
                low_spo2_threshold = 94
                if len(daily_mean_spo2) >= 2:
                    if all(daily_mean_spo2.iloc[-2:] < low_spo2_threshold):
                        pattern_alerts.append(f"Pattern Alert: Mean daily SpO2 has been below {low_spo2_threshold}% for the last 2 days.")
        except Exception as e:
            print(f"[CloudProcessor] Error during pattern recognition simulation: {e}")
        
        # --- Generate & Return Report Data ---
        report_data = self.generate_report_data(trends, baseline, pattern_alerts, health_df)
        return report_data  # Return structured data instead of printing directly

    def generate_report_data(self, trends, baseline, pattern_alerts, df):
        """Generates structured report data."""
        report = {
            "user_id": self.user_id,
            "report_generated_at": datetime.now().isoformat(),
            "total_data_points_processed": len(df),
            "baseline": baseline if baseline else "Not enough data",
            "trends": {},
            "ema": {},
            "pattern_alerts": pattern_alerts if pattern_alerts else ["No specific patterns flagged"]
        }
        
        if 'hourly' in trends and not trends['hourly'].empty:
            trends_hourly_df = trends['hourly'].copy()
            trends_hourly_df.columns = [' '.join(col).strip().replace('_', ' ').title() for col in trends_hourly_df.columns.values]
            # Convert index to string for JSON compatibility
            trends_hourly_df.index = trends_hourly_df.index.strftime('%Y-%m-%d %H:%M')
            report["trends"]["hourly_latest"] = trends_hourly_df.tail().to_dict()  # Send last few hours
        
        elif 'daily' in trends and not trends['daily'].empty:
            trends_daily_df = trends['daily'].copy()
            trends_daily_df.columns = [' '.join(col).strip().replace('_', ' ').title() for col in trends_daily_df.columns.values]
            trends_daily_df.index = trends_daily_df.index.strftime('%Y-%m-%d')
            report["trends"]["daily_latest"] = trends_daily_df.tail().to_dict()  # Send last few days
        
        if 'hr_ema_latest' in trends:
            report["ema"]["hr_latest"] = trends['hr_ema_latest']
        
        # Print the report to console as well for monitoring
        print("\n" + "="*50)
        print(f"🩺 Health Trend Report Generated for {self.user_id} 🩺")
        print(f"  Time: {report['report_generated_at']}")
        print(f"  Points Processed: {report['total_data_points_processed']}")
        print(f"  Baseline: {report['baseline']}")
        print(f"  Patterns: {report['pattern_alerts']}")
        print("\n" + "="*50)
        return report

def predict_risk(heart_rate, systolic_bp, diastolic_bp, body_temp, oxygen_sat):
    # Load the model and scaler
    try:
        model = joblib.load('risk_prediction_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        
        # Prepare the input data
        patient_data = [[heart_rate, systolic_bp, diastolic_bp, body_temp, oxygen_sat]]
        patient_data_scaled = scaler.transform(patient_data)
        
        # Make prediction
        prediction = model.predict(patient_data_scaled)
        probability = model.predict_proba(patient_data_scaled)
        # Convert numeric prediction back to label
        risk_labels = {v: k for k, v in risk_mapping.items()}
        predicted_label = risk_labels[prediction[0]]
        
        return {
            'risk_category': predicted_label,
            'probability': probability[0][list(model.classes_).index(prediction[0])]
        }
    except Exception as e:
        print(f"Error in risk prediction: {e}")
        return {
            'risk_category': 'Unknown',
            'probability': 0.0
        }

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes and origins

# Initialize database
setup_database()

# Instantiate Cloud Processor
cloud_processor = CloudProcessor(
    user_id="User_001",  # Hardcoded for now, could be dynamic
    ml_features=ML_FEATURES
)

# --- MQTT Callbacks for Flask ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("FLASK APP: Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC_SUB)
        print(f"FLASK APP: Subscribed to topic '{MQTT_TOPIC_SUB}'")
    else:
        print(f"FLASK APP: Failed to connect to MQTT, return code {rc}")

def on_message(client, userdata, msg):
    print(f"FLASK APP: Received message on topic '{msg.topic}'")
    try:
        payload_str = msg.payload.decode("utf-8")
        vitals = json.loads(payload_str)
        # Basic validation
        if all(key in vitals for key in ['timestamp', 'user_id'] + ML_FEATURES):
            # Pass data to the processor instance
            cloud_processor.process_live_data(vitals)
        else:
            print(f"FLASK APP: Received incomplete message: {payload_str}")
    except json.JSONDecodeError:
        print(f"FLASK APP: Failed to decode JSON: {msg.payload.decode('utf-8')}")
    except Exception as e:
        print(f"FLASK APP: Error processing message: {e}")

def on_disconnect(client, userdata, rc):
    print(f"FLASK APP: Disconnected from MQTT Broker (rc: {rc}). Attempting to reconnect...")

# --- Flask Routes ---
@app.route('/')
def index():
    with cloud_processor._lock:  # Access data count safely
        count = cloud_processor.data_count
    
    # Get count from database too
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM vital_signs WHERE user_id = '{cloud_processor.user_id}'")
        db_count = cursor.fetchone()[0]
        conn.close()
    except Exception as e:
        print(f"Error querying database count: {e}")
        db_count = "Error"
        
    return jsonify({
        "status": "Flask MQTT Processor Running",
        "user_id": cloud_processor.user_id,
        "data_points_received": count,
        "data_points_in_database": db_count
    })

@app.route('/report', methods=['GET'])
def get_report():
    """Returns an Excel file containing all vital signs for a specific user."""
    try:
        user_id = request.args.get('user_id', cloud_processor.user_id)
        print(f"FLASK APP: Excel report generation requested for user {user_id}")
        
        # Connect to database and fetch all vital signs for the user
        conn = sqlite3.connect(DB_PATH)
        query = f"SELECT * FROM vital_signs WHERE user_id = '{user_id}' ORDER BY timestamp"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return jsonify({"error": f"No data found for user {user_id}"}), 404
        
        # Create a temporary file for the Excel data
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            excel_path = temp_file.name
            
        # Write the dataframe to Excel
        df.to_excel(excel_path, index=False, sheet_name=f"Vitals_{user_id}")
        
        # Return the Excel file as a download
        return send_file(
            excel_path,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f"vital_signs_report_{user_id}.xlsx"
        )
    except Exception as e:
        print(f"Error generating Excel report: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/data', methods=['GET'])
def get_data():
    """Retrieve patient data from database"""
    try:
        user_id = request.args.get('user_id', cloud_processor.user_id)
        limit = request.args.get('limit', 100)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT * FROM vital_signs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", 
                      (user_id, limit))
        rows = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Convert to list of dictionaries
        data = [dict(zip(column_names, row)) for row in rows]
        
        conn.close()
        return jsonify({"data": data})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Retrieve alerts from database"""
    try:
        user_id = request.args.get('user_id', cloud_processor.user_id)
        limit = request.args.get('limit', 50)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT * FROM alerts WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", 
                      (user_id, limit))
        rows = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Convert to list of dictionaries
        alerts = [dict(zip(column_names, row)) for row in rows]
        
        conn.close()
        return jsonify({"alerts": alerts})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/latest', methods=['GET'])
def get_latest_data():
    """Retrieve the latest vital signs data for a user"""
    try:
        user_id = request.args.get('user_id', cloud_processor.user_id)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM vital_signs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (user_id,))
        row = cursor.fetchone()
        
        if not row:
            return jsonify({"error": f"No data found for user {user_id}"}), 404
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Convert to dictionary
        latest_data = dict(zip(column_names, row))
        
        conn.close()
        return jsonify({"latest_data": latest_data})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/latest_alert', methods=['GET'])
def get_latest_alert():
    """Retrieve the latest alert for a user"""
    try:
        user_id = request.args.get('user_id', cloud_processor.user_id)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM alerts WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (user_id,))
        row = cursor.fetchone()
        
        if not row:
            return jsonify({"error": f"No alerts found for user {user_id}"}), 404
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Convert to dictionary
        latest_alert = dict(zip(column_names, row))
        
        conn.close()
        return jsonify({"latest_alert": latest_alert})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/weekly_trends', methods=['GET'])
def get_weekly_trends():
    """Retrieve weekly health trends data for visualization"""
    try:
        user_id = request.args.get('user_id', cloud_processor.user_id)
        
        # Calculate date 7 days ago
        today = datetime.now()
        seven_days_ago = (today - pd.Timedelta(days=7)).isoformat()
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        
        # Query for the last 7 days of data
        query = """
        SELECT 
            date(timestamp) as date,
            avg(heart_rate) as heart_rate, 
            avg(systolic_bp) as systolic, 
            avg(diastolic_bp) as diastolic,
            avg(temperature) as temperature,
            avg(oxygen_saturation) as oxygen,
            avg(activity_level) as steps
        FROM vital_signs 
        WHERE user_id = ? AND timestamp >= ?
        GROUP BY date(timestamp)
        ORDER BY date(timestamp)
        """
        
        df = pd.read_sql_query(query, conn, params=(user_id, seven_days_ago))
        conn.close()
        
        if df.empty:
            return jsonify({"error": "No data available for the past week"}), 404
        
        # Format dates (Apr 6, Apr 7, etc.) with "Today" for the latest date
        dates = []
        latest_date = max(df['date']) if not df.empty else None
        
        for date in df['date']:
            if date == latest_date:
                dates.append("Today")
            else:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
                dates.append(date_obj.strftime('%b %d'))
        
        # Create the response data structure
        response = {
            "dates": dates,
            "heartRate": df['heart_rate'].round(1).tolist(),
            "systolic": df['systolic'].round(1).tolist(),
            "diastolic": df['diastolic'].round(1).tolist(),
            "temperature": df['temperature'].round(1).tolist(),
            "oxygen": df['oxygen'].round(1).tolist(),
            "steps": df['steps'].fillna(0).round(0).astype(int).tolist()  # Convert steps to integers
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error fetching weekly trends: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/user', methods=['GET'])
def get_user_data():
    """Retrieve user profile data for a specific user"""
    try:
        user_id = request.args.get('user_id', cloud_processor.user_id)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        
        if not row:
            return jsonify({"error": f"No user profile found for user {user_id}"}), 404
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Convert to dictionary
        user_data = dict(zip(column_names, row))
        
        conn.close()
        return jsonify({"user_data": user_data})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/user', methods=['POST'])
def create_or_update_user():
    """Create or update user profile data"""
    try:
        data = request.get_json()
        
        if not data or 'user_id' not in data:
            return jsonify({"error": "Missing required user_id field"}), 400
            
        user_id = data.get('user_id')
        username = data.get('username')
        password = data.get('password')
        name = data.get('name')
        height = data.get('height')
        weight = data.get('weight')
        
        # Calculate BMI if height and weight are provided
        bmi = None
        if height and weight and float(height) > 0:
            # BMI formula: weight(kg) / (height(m))^2
            height_m = float(height) / 100  # convert cm to meters if height is in cm
            weight_kg = float(weight)
            bmi = round(weight_kg / (height_m * height_m), 2)
        
        now = datetime.now().isoformat()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            # Update existing user
            cursor.execute('''
            UPDATE users 
            SET username = ?, password = ?, name = ?, height = ?, weight = ?, bmi = ?, updated_at = ?
            WHERE user_id = ?
            ''', (username, password, name, height, weight, bmi, now, user_id))
            message = "User updated successfully"
        else:
            # Create new user
            cursor.execute('''
            INSERT INTO users (user_id, username, password, name, height, weight, bmi, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, username, password, name, height, weight, bmi, now, now))
            message = "User created successfully"
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": message,
            "user_id": user_id,
            "bmi": bmi
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login_user():
    """Authenticate a user with username and password"""
    try:
        data = request.get_json()
        
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({"error": "Missing username or password"}), 400
            
        username = data.get('username')
        password = data.get('password')
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
                      (username, password))
        row = cursor.fetchone()
        
        if not row:
            return jsonify({"error": "Invalid username or password"}), 401
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Convert to dictionary
        user_data = dict(zip(column_names, row))
        
        # Don't return password in response
        if 'password' in user_data:
            del user_data['password']
        
        conn.close()
        return jsonify({
            "message": "Login successful",
            "user": user_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/risk_levels', methods=['GET'])
def get_risk_levels():
    """Get risk prediction for today, this week, and this month based on average vital signs"""
    try:
        user_id = request.args.get('user_id', cloud_processor.user_id)
        
        # Calculate date ranges
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        week_start = (now - pd.Timedelta(days=7)).isoformat()
        month_start = (now - pd.Timedelta(days=30)).isoformat()
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        
        # Query to get average vital signs for different time periods
        query = """
        SELECT 
            avg(heart_rate) as heart_rate, 
            avg(systolic_bp) as systolic_bp, 
            avg(diastolic_bp) as diastolic_bp,
            avg(temperature) as temperature,
            avg(oxygen_saturation) as oxygen_saturation
        FROM vital_signs 
        WHERE user_id = ? AND timestamp >= ?
        """
        
        # Get data for each time period
        today_data = pd.read_sql_query(query, conn, params=(user_id, today_start))
        week_data = pd.read_sql_query(query, conn, params=(user_id, week_start))
        month_data = pd.read_sql_query(query, conn, params=(user_id, month_start))
        
        conn.close()
        
        # Initialize response
        response = {
            "today": {"status": "no data"},
            "week": {"status": "no data"},
            "month": {"status": "no data"}
        }
        
        # Process today's data
        if not today_data.empty and not today_data['heart_rate'].isna().all():
            hr = float(today_data['heart_rate'].iloc[0] or 0)
            sys_bp = float(today_data['systolic_bp'].iloc[0] or 0)
            dia_bp = float(today_data['diastolic_bp'].iloc[0] or 0)
            temp = float(today_data['temperature'].iloc[0] or 0)
            spo2 = float(today_data['oxygen_saturation'].iloc[0] or 0)
            
            if hr > 0 and sys_bp > 0 and dia_bp > 0 and temp > 0 and spo2 > 0:
                risk_today = predict_risk(hr, sys_bp, dia_bp, temp, spo2)
                response["today"] = {
                    "risk_category": risk_today['risk_category'],
                    "probability": round(risk_today['probability'], 3),
                    "vitals": {
                        "heart_rate": round(hr, 1),
                        "systolic_bp": round(sys_bp, 1),
                        "diastolic_bp": round(dia_bp, 1),
                        "temperature": round(temp, 1),
                        "oxygen_saturation": round(spo2, 1)
                    }
                }
        
        # Process week data
        if not week_data.empty and not week_data['heart_rate'].isna().all():
            hr = float(week_data['heart_rate'].iloc[0] or 0)
            sys_bp = float(week_data['systolic_bp'].iloc[0] or 0)
            dia_bp = float(week_data['diastolic_bp'].iloc[0] or 0)
            temp = float(week_data['temperature'].iloc[0] or 0)
            spo2 = float(week_data['oxygen_saturation'].iloc[0] or 0)
            
            if hr > 0 and sys_bp > 0 and dia_bp > 0 and temp > 0 and spo2 > 0:
                risk_week = predict_risk(hr, sys_bp, dia_bp, temp, spo2)
                response["week"] = {
                    "risk_category": risk_week['risk_category'],
                    "probability": round(risk_week['probability'], 3),
                    "vitals": {
                        "heart_rate": round(hr, 1),
                        "systolic_bp": round(sys_bp, 1),
                        "diastolic_bp": round(dia_bp, 1),
                        "temperature": round(temp, 1),
                        "oxygen_saturation": round(spo2, 1)
                    }
                }
        
        # Process month data
        if not month_data.empty and not month_data['heart_rate'].isna().all():
            hr = float(month_data['heart_rate'].iloc[0] or 0)
            sys_bp = float(month_data['systolic_bp'].iloc[0] or 0)
            dia_bp = float(month_data['diastolic_bp'].iloc[0] or 0)
            temp = float(month_data['temperature'].iloc[0] or 0)
            spo2 = float(month_data['oxygen_saturation'].iloc[0] or 0)
            
            if hr > 0 and sys_bp > 0 and dia_bp > 0 and temp > 0 and spo2 > 0:
                risk_month = predict_risk(hr, sys_bp, dia_bp, temp, spo2)
                response["month"] = {
                    "risk_category": risk_month['risk_category'],
                    "probability": round(risk_month['probability'], 3),
                    "vitals": {
                        "heart_rate": round(hr, 1),
                        "systolic_bp": round(sys_bp, 1),
                        "diastolic_bp": round(dia_bp, 1),
                        "temperature": round(temp, 1),
                        "oxygen_saturation": round(spo2, 1)
                    }
                }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error calculating risk levels: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/ai_analysis', methods=['POST'])
def ai_health_analysis():
    """
    Endpoint to analyze health data using a medical-trained large language model.
    Expects a JSON payload with user data, current health metrics, historical data, and risk levels.
    Returns AI-generated health analysis and recommendations.
    """
    try:
        data = request.json
        
        # Validate required data
        if not data or 'user_data' not in data or 'current_health' not in data:
            return jsonify({'error': 'Missing required data'}), 400
        
        # Extract data for analysis
        user_data = data.get('user_data', {})
        current_health = data.get('current_health', {})
        historical_data = data.get('historical_data', [])
        risk_levels = data.get('risk_levels', {})
        
        # Prepare prompt for the AI model
        prompt = f"""
        You are a medical AI assistant trained to analyze health data and provide personalized medical advice.
        Please analyze the following health data and provide a comprehensive health assessment and recommendations.
        
        User Information:
        - Age: {user_data.get('age', 'Unknown')}
        - Gender: {user_data.get('gender', 'Unknown')}
        - Medical History: {user_data.get('medical_history', [])}
        - Medications: {user_data.get('medications', [])}
        
        Current Health Metrics:
        - Heart Rate: {current_health.get('heart_rate', 'Unknown')} BPM
        - Blood Pressure: {current_health.get('blood_pressure', {}).get('systolic', 'Unknown')}/{current_health.get('blood_pressure', {}).get('diastolic', 'Unknown')} mmHg
        - Body Temperature: {current_health.get('temperature', 'Unknown')} °C
        - Oxygen Saturation: {current_health.get('oxygen_saturation', 'Unknown')}%
        
        Historical Data:
        {json.dumps(historical_data, indent=2)}
        
        Risk Levels:
        - Today: {risk_levels.get('today', 'Unknown')}
        - Weekly: {risk_levels.get('weekly', 'Unknown')}
        - Monthly: {risk_levels.get('monthly', 'Unknown')}
        
        Please provide a comprehensive health analysis with the following sections:
        1. Overall Health Assessment
        2. Health Metrics Analysis
        3. Risk Assessment
        4. Recommendations
        5. Lifestyle Suggestions
        
        Format your response as a JSON object with these exact keys: overallAssessment, metricsAnalysis, riskAssessment, recommendations, lifestyleSuggestions, for each of them only should have text but not array, response without ```json
        """
        
        # Get OpenAI settings from the request payload
        openai_settings = data.get('openai_settings', {})
        api_endpoint = openai_settings.get('api_endpoint', 'https://api.james913.xyz')
        api_key = openai_settings.get('api_key', OPENAI_API_KEY)
        model = openai_settings.get('model', 'deepseek-chat')
        
        # Call AI API using requests instead of OpenAI client
        try:
            # Prepare headers and payload
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a medical AI assistant trained to analyze health data and provide personalized medical advice."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            # Make request to the API endpoint
            response = requests.post(
                f"{api_endpoint}/chat/completions",
                headers=headers,
                json=payload
            )
            
            # Check if request was successful
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract content from the API response
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    ai_response = response_data["choices"][0].get("message", {}).get("content", "")
                else:
                    ai_response = response_data.get("content", "")
            else:
                raise Exception(f"API request failed with status code: {response.status_code}, response: {response.text}")
            
            # Parse the JSON response
            try:
                analysis_result = json.loads(ai_response)
            except json.JSONDecodeError:
                # If the response is not valid JSON, create a structured response
                analysis_result = {
                    "overallAssessment": "Unable to parse AI response. Raw response: " + ai_response[:500],
                    "metricsAnalysis": "Please try again later.",
                    "riskAssessment": "Please try again later.",
                    "recommendations": "Please try again later.",
                    "lifestyleSuggestions": "Please try again later."
                }
            
            return jsonify(analysis_result)
            
        except Exception as e:
            # Fallback to a mock response if the API call fails
            print(f"Error calling AI API: {str(e)}")
            return jsonify({
                "overallAssessment": "Based on your health data, your overall health appears to be in good condition. Your vital signs are within normal ranges, and your activity level is moderate.",
                "metricsAnalysis": "Your heart rate is within the normal range. Your blood pressure readings are healthy. Your body temperature is normal. Your oxygen saturation is excellent, indicating good respiratory function.",
                "riskAssessment": "Your current risk level is low. There are no immediate health concerns based on the data provided. Continue monitoring your health metrics regularly.",
                "recommendations": "1. Continue your current exercise routine. 2. Maintain a balanced diet. 3. Ensure adequate sleep (7-9 hours per night). 4. Stay hydrated throughout the day.",
                "lifestyleSuggestions": "Consider incorporating more cardiovascular exercises into your routine. Try meditation or mindfulness practices to reduce stress. Focus on maintaining a consistent sleep schedule."
            })
            
    except Exception as e:
        print(f"Error in AI analysis: {str(e)}")
        return jsonify({'error': 'Failed to analyze health data'}), 500


# --- Main Execution ---
if __name__ == '__main__':
    # Setup MQTT Client for Flask
    mqtt_client = mqtt.Client(client_id=f"flask_processor_{cloud_processor.user_id}")
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.on_disconnect = on_disconnect
    
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"FLASK APP: CRITICAL - Could not connect to MQTT Broker on startup: {e}")
        # Decide if app should exit or keep trying
        exit(1)  # Exit if MQTT connection fails initially
    
    # Start MQTT loop in a background thread.
    mqtt_client.loop_start()
    print("FLASK APP: MQTT client loop started in background thread.")
    
    # Start the Flask development server
    print(f"FLASK APP: Starting Flask server on port {FLASK_PORT}...")
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=False)  # Set debug=False for cleaner logs
    
    # Cleanup when Flask server stops (e.g., Ctrl+C)
    print("FLASK APP: Stopping MQTT client loop...")
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("FLASK APP: MQTT client disconnected. Exiting.")
