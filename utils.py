import numpy as np

def calculate_efficiency(pressure, current):
    mass_flow = np.random.normal(0.5, 0.1, len(pressure))
    delta_h = np.random.normal(100000, 10000, len(pressure))
    voltage = 400
    power_factor = 0.9
    P_out = mass_flow * delta_h
    P_in = voltage * current * power_factor
    efficiency = np.clip((P_out / P_in) * 100, 85, 92)
    return efficiency

def calculate_vibration_amplitude(vibration):
    rotor_mass = 10
    angular_frequency = 2 * np.pi * 50
    kinetic_energy = 0.5 * rotor_mass * (vibration / 1000) ** 2
    amplitude = np.sqrt(2 * kinetic_energy / (rotor_mass * angular_frequency ** 2)) * 1000
    return np.clip(amplitude, 0, 1)