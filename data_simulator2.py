import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import calculate_efficiency, calculate_vibration_amplitude
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Realistic screw compressor fault thresholds
SCREW_FAULTS = {
    'High Vibration': {'condition': lambda x: x['Vibration (mm/s)'] > 6, 'cause': 'Possible bearing wear or misalignment'},
    'Overheating': {'condition': lambda x: x['Temperature (°C)'] > 110, 'cause': 'Cooling system failure or high load'},
    'Low Efficiency': {'condition': lambda x: x['Efficiency (%)'] < 80, 'cause': 'Air leakage or oil contamination'},
    'High Current': {'condition': lambda x: x['Current (A)'] > 50, 'cause': 'Electrical overload or motor issue'}
}

# Comprehensive list of suggested checks, categorized
SUGGESTED_CHECKS = {
    'Electrical': [
        "Are the contactors in good condition? (Check for wear, arcing, or overheating.)",
        "Are all fuses okay? (Inspect for blown fuses or incorrect ratings.)",
        "Are the phase connections correct? (Verify phase sequence and balance.)",
        "Is the motor insulation resistance within acceptable limits? (Check using a megger test.)",
        "Are there any signs of loose electrical connections? (Inspect terminals for heat marks or looseness.)",
        "Is the control panel wiring intact? (Look for frayed or damaged wires.)",
        "Is the motor current showing signs of imbalance? (Recommend a phase-by-phase current analysis.)"
    ],
    'Mechanical': [
        "Is the bearing misaligned? (Check alignment using a laser alignment tool.)",
        "Are there signs of bearing wear or damage? (Listen for unusual noises or check vibration patterns.)",
        "Is the rotor bar in good condition? (Inspect for cracks or wear in the rotor assembly.)",
        "Are the compressor seals intact? (Check for oil or air leaks around seals.)",
        "Is the coupling between the motor and compressor properly aligned? (Verify coupling condition.)",
        "Has there been a recent spike in vibration trends? (Analyze historical vibration data for patterns.)"
    ],
    'Pneumatic/Hydraulic': [
        "Is there any hose leakage? (Inspect all hoses for cracks, wear, or loose fittings.)",
        "Is the air filter clogged or dirty? (Check for restricted airflow impacting efficiency.)",
        "Is the oil level within the recommended range? (Verify oil reservoir levels.)",
        "Is the oil separator functioning correctly? (Check for excessive oil carryover in the discharge air.)",
        "Are the pressure relief valves operational? (Test to ensure they open at the correct pressure.)",
        "Are there signs of water contamination in the oil? (Inspect oil for emulsification or cloudiness.)",
        "Is the efficiency drop correlated with load changes? (Suggest optimizing operational load.)"
    ],
    'Environmental/Operational': [
        "Is the ambient temperature within the compressor’s operating range? (High temperatures can cause overheating.)",
        "Is there excessive dust or debris around the compressor? (Check for blockages in cooling fins or air intakes.)",
        "Is the compressor operating under excessive load? (Verify load against rated capacity.)",
        "Is the cooling system (fan, radiator, or water-cooling) functioning properly? (Check for blockages or pump failures.)",
        "Is the temperature trending upward over time? (Recommend checking cooling system efficiency.)",
        "Are there recurring fault patterns in this compressor? (Review historical fault logs for root cause analysis.)"
    ]
}

# Mapping of fault types to prioritized check categories
FAULT_TO_CHECK_CATEGORIES = {
    'High Vibration': ['Mechanical', 'Environmental/Operational'],
    'Overheating': ['Environmental/Operational', 'Pneumatic/Hydraulic'],
    'Low Efficiency': ['Pneumatic/Hydraulic', 'Environmental/Operational'],
    'High Current': ['Electrical', 'Mechanical'],
    'None': []  # No checks for 'None'
}

# State for dynamic updates with realistic operating ranges
last_values = {i: {'pressure': 10, 'temperature': 50, 'vibration': 2, 'current': 30, 'efficiency': 90} for i in range(10)}
fault_applied = {i: None for i in range(10)}  # Track if a fault is applied to persist the fault state

# Assign unique IDs to compressors
COMPRESSOR_IDS = {i: f"COMP-{str(i+1).zfill(3)}" for i in range(10)}  # e.g., COMP-001 to COMP-010

# Q-learning state for maintenance scheduling
q_table = {i: np.zeros((2, 2)) for i in range(10)}  # State: (failure_prob_high, action), Actions: (schedule_now, wait)
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Pre-train LSTM model
sequence_length = 5  # Reduced sequence length for faster processing
scaler = MinMaxScaler()
lstm_model = None

def build_lstm_model(sequence_length, n_features):
    """Build and compile an LSTM model for failure prediction."""
    model = Sequential([
        LSTM(32, activation='tanh', input_shape=(sequence_length, n_features)),  # Simplified model
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def prepare_lstm_data(df, sequence_length=5):
    """Prepare data for LSTM by creating sequences."""
    features = ['Vibration (mm/s)', 'Temperature (°C)', 'Efficiency (%)', 'Current (A)']
    scaled_data = scaler.transform(df[features])
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        next_row = df.iloc[i + sequence_length]
        fault_detected = any(fault_info['condition'](next_row) for fault_info in SCREW_FAULTS.values())
        y.append(1 if fault_detected else 0)
    
    return np.array(X), np.array(y)

def initialize_lstm_model():
    global lstm_model, scaler
    # Generate synthetic data for initial training
    synthetic_data = []
    for comp_id in range(10):
        data = []
        for i in range(50):  # Generate 50 points for initial training
            pressure = np.random.uniform(8, 12)
            temperature = np.random.uniform(40, 70)
            vibration = np.random.uniform(1, 4)
            current = np.random.uniform(25, 40)
            efficiency = np.random.uniform(85, 95)
            # Inject some faults for training
            if i % 10 == 0:  # Every 10th point, inject a fault
                fault_type = np.random.choice(['High Vibration', 'Overheating', 'Low Efficiency', 'High Current'])
                if fault_type == 'High Vibration':
                    vibration = np.random.uniform(7, 10)
                elif fault_type == 'Overheating':
                    temperature = np.random.uniform(120, 150)
                elif fault_type == 'Low Efficiency':
                    efficiency = np.random.uniform(70, 75)
                elif fault_type == 'High Current':
                    current = np.random.uniform(55, 60)
            data.append({
                'Compressor ID': comp_id,
                'Pressure (bar)': pressure,
                'Temperature (°C)': temperature,
                'Vibration (mm/s)': vibration,
                'Current (A)': current,
                'Efficiency (%)': efficiency
            })
        synthetic_data.append(pd.DataFrame(data))
    synthetic_df = pd.concat(synthetic_data, ignore_index=True)

    # Prepare data for LSTM
    features = ['Vibration (mm/s)', 'Temperature (°C)', 'Efficiency (%)', 'Current (A)']
    scaler.fit(synthetic_df[features])
    X, y = prepare_lstm_data(synthetic_df, sequence_length)

    # Build and train the model
    if len(X) > 0:
        lstm_model = build_lstm_model(sequence_length, X.shape[2])
        lstm_model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        print("LSTM model pre-trained successfully")

# Call the initialization at startup
initialize_lstm_model()

def generate_sensor_data(num_compressors=10, n_points=1, fault_injection=None):
    """
    Generate realistic sensor data for compressors with unique IDs.
    fault_injection: dict with keys as compressor IDs (0-9) and values as fault types
    """
    global last_values, fault_applied
    np.random.seed(42)
    time = [datetime.now() - timedelta(seconds=(n_points-i-1)*5) for i in range(n_points)]
    
    all_data = []
    for comp_id in range(num_compressors):
        data_points = []
        for t in time:
            pressure = last_values[comp_id]['pressure']
            temperature = last_values[comp_id]['temperature']
            vibration = last_values[comp_id]['vibration']
            current = last_values[comp_id]['current']
            efficiency = last_values[comp_id]['efficiency']

            if fault_injection and comp_id in fault_injection:
                fault_type = fault_injection[comp_id]
                fault_applied[comp_id] = fault_type
                print(f"Injecting {fault_type} into Compressor {comp_id} ({COMPRESSOR_IDS[comp_id]}) at time {t}")
                if fault_type == 'High Vibration':
                    vibration = np.random.uniform(7, 10)
                    print(f"Applied High Vibration: Vibration (mm/s) set to {vibration}")
                elif fault_type == 'Overheating':
                    temperature = np.random.uniform(120, 150)
                    print(f"Applied Overheating: Temperature (°C) set to {temperature}")
                elif fault_type == 'Low Efficiency':
                    efficiency = np.random.uniform(70, 75)
                    print(f"Applied Low Efficiency: Efficiency (%) set to {efficiency}")
                elif fault_type == 'High Current':
                    current = np.random.uniform(55, 60)
                    print(f"Applied High Current: Current (A) set to {current}")
            elif fault_applied[comp_id]:
                fault_type = fault_applied[comp_id]
                print(f"Persisting {fault_type} for Compressor {comp_id} ({COMPRESSOR_IDS[comp_id]}) at time {t}")
                if fault_type == 'High Vibration':
                    vibration = np.random.uniform(7, 10)
                    print(f"Persisted High Vibration: Vibration (mm/s) set to {vibration}")
                elif fault_type == 'Overheating':
                    temperature = np.random.uniform(120, 150)
                    print(f"Persisted Overheating: Temperature (°C) set to {temperature}")
                elif fault_type == 'Low Efficiency':
                    efficiency = np.random.uniform(70, 75)
                    print(f"Persisted Low Efficiency: Efficiency (%) set to {efficiency}")
                elif fault_type == 'High Current':
                    current = np.random.uniform(55, 60)
                    print(f"Persisted High Current: Current (A) set to {current}")
            else:
                if np.random.random() < 0.01:
                    anomaly_type = np.random.choice(['High Vibration', 'Overheating', 'Low Efficiency', 'High Current'])
                    print(f"Natural anomaly {anomaly_type} in Compressor {comp_id} ({COMPRESSOR_IDS[comp_id]}) at time {t}")
                    if anomaly_type == 'High Vibration':
                        vibration = np.random.uniform(7, 10)
                    elif anomaly_type == 'Overheating':
                        temperature = np.random.uniform(120, 150)
                    elif anomaly_type == 'Low Efficiency':
                        efficiency = np.random.uniform(70, 75)
                    elif anomaly_type == 'High Current':
                        current = np.random.uniform(55, 60)
                else:
                    pressure = np.clip(pressure + np.random.normal(0, 0.2), 8, 12)
                    temperature = np.clip(temperature + np.random.normal(0, 0.5), 40, 70)
                    vibration = np.clip(vibration + np.random.normal(0, 0.05), 1, 4)
                    current = np.clip(current + np.random.normal(0, 0.3), 25, 40)
                    efficiency = np.clip(efficiency + np.random.normal(0, 0.2), 85, 95)

            last_values[comp_id] = {
                'pressure': pressure,
                'temperature': temperature,
                'vibration': vibration,
                'current': current,
                'efficiency': efficiency
            }

            traditional_efficiency = np.clip(efficiency - np.random.uniform(5, 10), 75, 80)
            vibration_amplitude_array = calculate_vibration_amplitude(np.array([vibration]))
            vibration_amplitude = float(vibration_amplitude_array[0]) if isinstance(vibration_amplitude_array, np.ndarray) else float(vibration_amplitude_array)

            data = {
                'Time': t,
                'Compressor ID': comp_id,
                'Unique ID': COMPRESSOR_IDS[comp_id],
                'Pressure (bar)': pressure,
                'Temperature (°C)': temperature,
                'Vibration (mm/s)': vibration,
                'Current (A)': current,
                'Efficiency (%)': efficiency,
                'Traditional Efficiency (%)': traditional_efficiency,
                'Vibration Amplitude (mm)': vibration_amplitude,
                'Fault Probability': 0.0
            }
            data_points.append(data)
        
        comp_data = pd.DataFrame(data_points)
        print(f"Generated data for Compressor {comp_id} ({COMPRESSOR_IDS[comp_id]}): {comp_data[['Vibration (mm/s)', 'Temperature (°C)', 'Efficiency (%)', 'Current (A)', 'Vibration Amplitude (mm)']].iloc[-1].to_dict()}")
        all_data.append(comp_data)
    
    return pd.concat(all_data, ignore_index=True)

def reset_fault_state():
    """Reset the fault state for all compressors."""
    global fault_applied, last_values
    fault_applied = {i: None for i in range(10)}
    last_values = {i: {'pressure': 10, 'temperature': 50, 'vibration': 2, 'current': 30, 'efficiency': 90} for i in range(10)}
    print("Fault state and last values reset for all compressors")

def get_suggested_checks(fault_type):
    """Return a string of suggested checks with serial numbers for the service engineer."""
    if fault_type == 'None':
        return "No additional checks required."
    
    categories = FAULT_TO_CHECK_CATEGORIES.get(fault_type, [])
    if not categories:
        return "No additional checks required."
    
    checks = []
    serial_num = 1
    for category in categories:
        for check in SUGGESTED_CHECKS[category]:
            checks.append(f"{serial_num}. {check}")
            serial_num += 1
    
    return "\n".join(checks)

def predict_with_lstm(df, sequence_length=5):
    """Predict failure probabilities using the pre-trained LSTM model."""
    global lstm_model, scaler
    features = ['Vibration (mm/s)', 'Temperature (°C)', 'Efficiency (%)', 'Current (A)']
    df_subset = df[features].copy()
    
    # Ensure we have enough data points
    if len(df_subset) < sequence_length:
        # Pad with the earliest data if not enough points
        padding = pd.DataFrame([df_subset.iloc[0]] * (sequence_length - len(df_subset)), columns=features)
        df_subset = pd.concat([padding, df_subset], ignore_index=True)
    
    scaled_data = scaler.transform(df_subset)
    
    X = []
    for i in range(len(scaled_data) - sequence_length + 1):
        X.append(scaled_data[i:i + sequence_length])
    
    X = np.array(X)
    if len(X) == 0:
        return np.zeros(len(df))
    
    probabilities = lstm_model.predict(X, verbose=0, batch_size=32)
    
    # Align probabilities with DataFrame
    padded_probabilities = np.zeros(len(df))
    padded_probabilities[sequence_length-1:] = probabilities.flatten()
    for i in range(sequence_length-1):
        padded_probabilities[i] = padded_probabilities[sequence_length-1] if len(padded_probabilities) > sequence_length-1 else 0
    
    return padded_probabilities

def q_learning_update(comp_id, failure_prob, action, reward):
    """Update the Q-table using Q-learning."""
    global q_table
    state = int(failure_prob > 0.7)
    current_q = q_table[comp_id][state, action]
    max_future_q = np.max(q_table[comp_id][state])
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
    q_table[comp_id][state, action] = new_q

def get_rl_action(comp_id, failure_prob):
    """Choose an action using epsilon-greedy policy."""
    state = int(failure_prob > 0.7)
    if np.random.random() < epsilon:
        return np.random.choice([0, 1])
    return np.argmax(q_table[comp_id][state])

def calculate_reward(failure_prob, action):
    """Calculate reward based on action and failure probability."""
    if action == 1:
        if failure_prob > 0.7:
            return 10
        return -5
    else:
        if failure_prob > 0.7:
            return -10
        return 2

def predict_failures(df):
    df = df.copy()
    df['Vibration Change'] = df.groupby('Compressor ID')['Vibration (mm/s)'].diff().fillna(0)
    df['Temperature Change'] = df.groupby('Compressor ID')['Temperature (°C)'].diff().fillna(0)
    df['Current Change'] = df.groupby('Compressor ID')['Current (A)'].diff().fillna(0)
    df['Efficiency Change'] = df.groupby('Compressor ID')['Efficiency (%)'].diff().fillna(0)

    # Predict failure probabilities using pre-trained LSTM
    probabilities = predict_with_lstm(df, sequence_length)
    df['Fault Probability'] = np.clip(probabilities, 0, 1)

    # Ensure fault probabilities reflect actual faults
    fault_reasons = np.array(['None'] * len(df), dtype=object)
    for fault_name, fault_info in SCREW_FAULTS.items():
        condition_result = fault_info['condition'](df)
        fault_reasons = np.where(condition_result, fault_info['cause'], fault_reasons)
        # Boost probability if a fault condition is met
        df['Fault Probability'] = np.where(condition_result, np.maximum(df['Fault Probability'], 0.95), df['Fault Probability'])
    
    df['Fault Reason'] = fault_reasons
    
    failures = df.groupby(['Compressor ID', 'Unique ID']).last().reset_index()

    reason_to_fault_type = {info['cause']: fault_name for fault_name, info in SCREW_FAULTS.items()}
    reason_to_fault_type['None'] = 'None'

    failures['Suggested Checks'] = failures['Fault Reason'].apply(
        lambda x: get_suggested_checks(reason_to_fault_type.get(x, 'None'))
    )

    for _, row in failures.iterrows():
        if row['Fault Reason'] != 'None':
            checks = get_suggested_checks(reason_to_fault_type.get(row['Fault Reason'], 'None'))
            print(f"Sending email to service engineer for {row['Unique ID']}: Suggested Checks:\n{checks}")

    failures['Maintenance Action'] = failures.apply(
        lambda row: get_rl_action(row['Compressor ID'], row['Fault Probability']), axis=1
    )
    failures['Action Description'] = failures['Maintenance Action'].map({0: 'Wait', 1: 'Schedule Now'})

    for _, row in failures.iterrows():
        comp_id = row['Compressor ID']
        action = row['Maintenance Action']
        reward = calculate_reward(row['Fault Probability'], action)
        q_learning_update(comp_id, row['Fault Probability'], action, reward)

    failures = failures[['Compressor ID', 'Unique ID', 'Fault Probability', 'Fault Reason', 'Suggested Checks', 'Action Description']]
    failures.columns = ['Compressor ID', 'Unique ID', 'Failure Probability', 'Tentative Reason', 'Suggested Checks', 'Action']
    # print(f"Failures DataFrame: {failures}")
    return df, failures