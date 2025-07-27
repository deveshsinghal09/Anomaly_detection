import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Setup
num_vehicles = 100
days = 30
start_date = datetime(2025, 7, 1)
vehicle_ids = [f"VEH-{i:03d}" for i in range(1, num_vehicles + 1)]

data = []

for day in range(days):
    date = start_date + timedelta(days=day)
    for vehicle in vehicle_ids:
        # Simulate normal values
        speed = np.clip(random.gauss(55, 15), 0, 130)
        engine_temp = np.clip(random.gauss(85, 10), 60, 130)
        fuel_efficiency = np.clip(random.gauss(15, 3), 5, 25)
        ambient_temp = random.randint(25, 40)
        rpm = random.randint(2000, 5000)
        brake_events = random.randint(0, 6)
        
        # Default values
        fault_code = None
        anomaly_type = "normal"

        # Inject Anomalies
        if engine_temp > 100 and speed < 30:
            anomaly_type = "engine_overheat_at_idle"
            fault_code = "P0128"
        elif fuel_efficiency < 8:
            anomaly_type = "low_fuel_efficiency"
        elif brake_events > 5:
            anomaly_type = "excessive_braking"
        elif rpm > 4800:
            anomaly_type = "high_rpm"

        data.append({
            "vehicle_id": vehicle,
            "timestamp": date.strftime("%Y-%m-%d"),
            "speed": round(speed, 2),
            "engine_temp": round(engine_temp, 2),
            "fuel_efficiency": round(fuel_efficiency, 2),
            "ambient_temp": ambient_temp,
            "rpm": rpm,
            "brake_events": brake_events,
            "fault_code": fault_code,
            "anomaly_type": anomaly_type
        })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("synthetic_vehicle_data_with_anomalies.csv", index=False)

print("Dataset generated and saved as synthetic_vehicle_data_with_anomalies.csv")
