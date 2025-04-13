import paho.mqtt.client as mqtt
import time
import json
import random
from datetime import datetime
import numpy as np

# --- Configuration ---
MQTT_BROKER = "test.mosquitto.org"  # Or IP address of your MQTT broker
MQTT_PORT = 1883
MQTT_TOPIC = "health/user_001/vitals" # Topic to publish sensor data

SIMULATION_DURATION_SECONDS = 1800 # Run for 3 minutes
DATA_INTERVAL_SECONDS = 2
USER_ID = "User_001"

# Sensor Simulation Parameters (copied from previous logic)
NORMAL_RANGES = {
    'heart_rate': (55, 100), 'systolic_bp': (90, 130), 'diastolic_bp': (60, 85),
    'temperature': (36.1, 37.5), 'oxygen_saturation': (95, 100), 'activity_level': (0, 10)
}
ALERT_THRESHOLDS = { # Used only for *generating* occasional test anomalies
    'heart_rate': (45, 120), 'systolic_bp': (85, 140), 'diastolic_bp': (55, 90),
    'temperature': (35.0, 38.5), 'oxygen_saturation': (92, 101),
}

# --- Sensor Data Simulation Function ---
def simulate_vitals(last_vitals=None):
    """Generates a single reading of simulated vital signs."""
    vitals = {}
    # Simulate slight variations (same logic as before)
    if last_vitals:
        hr = last_vitals['heart_rate'] + random.uniform(-2, 2)
        sys_bp = last_vitals['systolic_bp'] + random.uniform(-3, 3)
        dia_bp = last_vitals['diastolic_bp'] + random.uniform(-2, 2)
        temp = last_vitals['temperature'] + random.uniform(-0.1, 0.1)
        spo2 = last_vitals['oxygen_saturation'] + random.uniform(-0.5, 0.5)
        activity = last_vitals['activity_level'] + random.randint(-1, 1)
        # Bias towards returning to normal if outside range (copied logic)
        if hr < NORMAL_RANGES['heart_rate'][0] or hr > NORMAL_RANGES['heart_rate'][1]: hr += (np.mean(NORMAL_RANGES['heart_rate']) - hr) * 0.1
        if sys_bp < NORMAL_RANGES['systolic_bp'][0] or sys_bp > NORMAL_RANGES['systolic_bp'][1]: sys_bp += (np.mean(NORMAL_RANGES['systolic_bp']) - sys_bp) * 0.1
        if dia_bp < NORMAL_RANGES['diastolic_bp'][0] or dia_bp > NORMAL_RANGES['diastolic_bp'][1]: dia_bp += (np.mean(NORMAL_RANGES['diastolic_bp']) - dia_bp) * 0.1
        if temp < NORMAL_RANGES['temperature'][0] or temp > NORMAL_RANGES['temperature'][1]: temp += (np.mean(NORMAL_RANGES['temperature']) - temp) * 0.1
        if spo2 < NORMAL_RANGES['oxygen_saturation'][0]: spo2 += (NORMAL_RANGES['oxygen_saturation'][0] - spo2) * 0.2
    else: # Initial values
        hr = random.uniform(60, 85)
        sys_bp = random.uniform(100, 125)
        dia_bp = random.uniform(65, 80)
        temp = random.uniform(36.5, 37.2)
        spo2 = random.uniform(96, 99)
        activity = random.randint(0, 3)

    # Simulate occasional realistic spikes/dips (same logic)
    if random.random() < 0.05: hr += random.uniform(-15, 15)
    if random.random() < 0.03:
        sys_bp += random.uniform(-10, 10)
        dia_bp += random.uniform(-7, 7)
    if activity > 6 and random.random() < 0.2:
         hr += random.uniform(5, 20) * (activity / 5)
         sys_bp += random.uniform(3, 8) * (activity / 5)

    # *** Use ISO format string for timestamp ***
    vitals['timestamp'] = datetime.now().isoformat() # Crucial for JSON!
    vitals['user_id'] = USER_ID
    vitals['heart_rate'] = round(max(40, min(180, hr)), 1)
    vitals['systolic_bp'] = round(max(70, min(190, sys_bp)), 0)
    vitals['diastolic_bp'] = round(max(40, min(120, dia_bp)), 0)
    vitals['temperature'] = round(max(34.0, min(41.0, temp)), 2)
    vitals['oxygen_saturation'] = round(max(85, min(100, spo2)), 1)
    vitals['activity_level'] = int(max(0, min(10, activity)))

    # Occasionally simulate an actual anomaly for testing alerts (same logic)
    if random.random() < 0.03: # 3% chance
        param_to_spike = random.choice(list(ALERT_THRESHOLDS.keys()))
        print(f"*** SIMULATOR: Intentionally generating anomaly potential in {param_to_spike}... ***")
        if random.random() < 0.5: # Spike high
             vitals[param_to_spike] = ALERT_THRESHOLDS[param_to_spike][1] + random.uniform(1, 10)
        else: # Spike low
             vitals[param_to_spike] = ALERT_THRESHOLDS[param_to_spike][0] - random.uniform(1, 10)
        # Clamp again after spike
        vitals['heart_rate'] = round(max(40, min(180, vitals['heart_rate'])), 1)
        vitals['systolic_bp'] = round(max(70, min(190, vitals['systolic_bp'])), 0)
        vitals['diastolic_bp'] = round(max(40, min(120, vitals['diastolic_bp'])), 0)
        vitals['temperature'] = round(max(34.0, min(41.0, vitals['temperature'])), 2)
        vitals['oxygen_saturation'] = round(max(85, min(100, vitals['oxygen_saturation'])), 1)

    return vitals

# --- MQTT Helper Functions ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("SIMULATOR: Connected to MQTT Broker!")
    else:
        print(f"SIMULATOR: Failed to connect, return code {rc}")

def on_disconnect(client, userdata, rc):
    print(f"SIMULATOR: Disconnected from MQTT Broker (rc: {rc}).")

# --- Main Simulation Loop ---
if __name__ == "__main__":
    print(f"--- Starting Sensor Simulator for User: {USER_ID} ---")
    print(f"--- Publishing to MQTT: {MQTT_BROKER}:{MQTT_PORT} on topic '{MQTT_TOPIC}' ---")

    client = mqtt.Client(client_id=f"sensor_simulator_{USER_ID}")
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"SIMULATOR: Error connecting to MQTT Broker: {e}")
        exit()

    client.loop_start() # Start network loop in background thread

    last_reading = None
    start_time = time.time()
    message_count = 0

    try:
        while time.time() - start_time < SIMULATION_DURATION_SECONDS:
            current_vitals = simulate_vitals(last_reading)
            last_reading = current_vitals

            # Convert data to JSON string
            payload = json.dumps(current_vitals)

            # Publish to MQTT
            result = client.publish(MQTT_TOPIC, payload)
            status = result.rc
            if status == mqtt.MQTT_ERR_SUCCESS:
                 print(f"[{datetime.now().strftime('%H:%M:%S')}] SIMULATOR: Sent data -> {payload}")
                 message_count += 1
            else:
                 print(f"SIMULATOR: Failed to send message to topic {MQTT_TOPIC} (rc: {status})")

            time.sleep(DATA_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("SIMULATOR: Simulation stopped by user.")
    finally:
        print(f"\n--- SIMULATOR: Simulation Finished ---")
        print(f"--- SIMULATOR: Sent {message_count} messages ---")
        client.loop_stop() # Stop network loop
        client.disconnect()
        print("SIMULATOR: Disconnected.")
