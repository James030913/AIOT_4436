import sqlite3
import random
from datetime import datetime, timedelta
import numpy as np

# Database configuration
DB_PATH = 'health_data.db'

# Normal ranges for vital signs
NORMAL_RANGES = {
    'heart_rate': (60, 100),
    'systolic_bp': (90, 120),
    'diastolic_bp': (60, 80),
    'temperature': (36.1, 37.0),
    'oxygen_saturation': (95, 100),
    'activity_level': (1000, 10000)  # steps per day
}

# Risk categories with probabilities for normal readings
RISK_CATEGORIES = ["Low", "Medium"]
RISK_PROBABILITIES = {
    "Low": (0.05, 0.3),
    "Medium": (0.3, 0.5)
}

def generate_normal_vitals():
    """Generate vital signs within normal ranges"""
    heart_rate = round(random.uniform(*NORMAL_RANGES['heart_rate']), 1)
    systolic_bp = round(random.uniform(*NORMAL_RANGES['systolic_bp']), 1)
    diastolic_bp = round(random.uniform(*NORMAL_RANGES['diastolic_bp']), 1)
    temperature = round(random.uniform(*NORMAL_RANGES['temperature']), 1)
    oxygen_saturation = round(random.uniform(*NORMAL_RANGES['oxygen_saturation']), 1)
    
    # Determine risk category - mostly low risk since values are normal
    risk_category = np.random.choice(RISK_CATEGORIES, p=[0.8, 0.2])
    risk_probability = round(random.uniform(*RISK_PROBABILITIES[risk_category]), 2)
    
    return {
        'heart_rate': heart_rate,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'temperature': temperature,
        'oxygen_saturation': oxygen_saturation,
        'risk_category': risk_category,
        'risk_probability': risk_probability
    }

def generate_activity_level(hour):
    """Generate activity level based on time of day"""
    # Lower activity during sleep hours (22:00-07:00)
    if hour < 7 or hour >= 22:
        return round(random.uniform(0, 500))
    # Higher activity during day hours
    elif 9 <= hour < 19:
        return round(random.uniform(500, 2000))
    # Moderate activity during transition hours
    else:
        return round(random.uniform(300, 1000))

def add_data_to_db():
    """Add 7 days of data to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # User ID
    user_id = "User_001"
    
    # Current time
    now = datetime.now()
    
    # Generate data for past 7 days, with readings every 2-3 hours
    for day in range(7, 0, -1):
        # Starting date (days ago)
        current_date = now - timedelta(days=day)
        
        # Generate 8-10 readings per day
        num_readings = random.randint(8, 10)
        
        # Determine total activity for the day (step count)
        daily_activity = random.randint(*NORMAL_RANGES['activity_level'])
        remaining_activity = daily_activity
        
        hours = sorted(random.sample(range(0, 24), num_readings))
        
        for i, hour in enumerate(hours):
            current_datetime = current_date.replace(hour=hour, minute=random.randint(0, 59))
            
            # Generate vital signs
            vitals = generate_normal_vitals()
            
            # Calculate activity for this time period
            if i == len(hours) - 1:
                # Last reading gets all remaining activity
                activity = remaining_activity
            else:
                # Distribute activity based on time of day
                activity_percent = generate_activity_level(hour) / 2000
                activity = min(round(daily_activity * activity_percent), remaining_activity)
                remaining_activity -= activity
            
            # Insert data into the database
            cursor.execute('''
            INSERT INTO vital_signs 
            (user_id, timestamp, heart_rate, systolic_bp, diastolic_bp, temperature, 
             oxygen_saturation, activity_level, risk_category, risk_probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                current_datetime.isoformat(),
                vitals['heart_rate'],
                vitals['systolic_bp'],
                vitals['diastolic_bp'],
                vitals['temperature'],
                vitals['oxygen_saturation'],
                activity,
                vitals['risk_category'],
                vitals['risk_probability']
            ))
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Successfully added 7 days of data for {user_id} to the database.")

if __name__ == "__main__":
    add_data_to_db()
