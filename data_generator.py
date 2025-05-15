import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_heat_transfer_dataset(n_samples=100, random_seed=42):
    np.random.seed(random_seed)

    # Generate timestamps
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]

    # Base parameters with realistic ranges
    time = np.linspace(0, n_samples, n_samples)

    # Input parameters
    inlet_temperature = 80 + 15 * np.sin(time / 20) + np.random.normal(0, 2, n_samples)
    outlet_temperature = inlet_temperature - (10 + 2 * np.sin(time / 15) + np.random.normal(0, 1, n_samples))

    flow_rate = 2.5 + 0.5 * np.sin(time / 30) + np.random.normal(0, 0.1, n_samples)
    pressure = 101.325 + 5 * np.sin(time / 25) + np.random.normal(0, 0.5, n_samples)

    # Calculate derived parameters
    fluid_density = 1000 * (1 - 0.000214 * (inlet_temperature - 4) ** 2)  # Simplified water density
    specific_heat = 4.186  # kJ/kg·K (water)

    # Heat transfer calculations
    temperature_difference = inlet_temperature - outlet_temperature
    heat_transfer_rate = fluid_density * flow_rate * specific_heat * temperature_difference

    # Surface area and heat flux
    surface_area = 2.5  # m²
    heat_flux = heat_transfer_rate / surface_area

    # Reynolds number (simplified)
    pipe_diameter = 0.05  # m
    fluid_velocity = flow_rate / (np.pi * (pipe_diameter / 2) ** 2)
    viscosity = 0.001  # Pa·s (water)
    reynolds_number = (fluid_density * fluid_velocity * pipe_diameter) / viscosity

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'inlet_temperature': inlet_temperature,
        'outlet_temperature': outlet_temperature,
        'flow_rate': flow_rate,
        'pressure': pressure,
        'fluid_density': fluid_density,
        'heat_transfer_rate': heat_transfer_rate,
        'heat_flux': heat_flux,
        'reynolds_number': reynolds_number,
        'efficiency': (heat_transfer_rate / (flow_rate * specific_heat * inlet_temperature)) * 100
    })

    # Add some noise to make it more realistic
    data['efficiency'] += np.random.normal(0, 1, n_samples)

    # Add operating conditions
    data['operating_mode'] = np.where(data['flow_rate'] > 2.7, 'high_flow',
                                      np.where(data['flow_rate'] < 2.3, 'low_flow', 'normal'))

    return data


# Generate the dataset
heat_transfer_data = generate_heat_transfer_dataset(100)

# Display the first few rows and basic statistics
print("First 5 rows of the dataset:")
print(heat_transfer_data.head())
print("\nDataset Statistics:")
print(heat_transfer_data.describe())

# Save to CSV
heat_transfer_data.to_csv('heat_transfer_data.csv', index=False)