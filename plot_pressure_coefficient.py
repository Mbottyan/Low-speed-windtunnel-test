import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the experimental data
experimental_data_path = 'data files/cp_data/experimental/cp_data at 10.2° aoa.csv'
experimental_data = pd.read_csv(experimental_data_path)

x_experimental = experimental_data['# x']
cp_experimental = experimental_data[' Cp']

"""# Load the simulation data
simulation_data_path = 'data files/cp_data/simulation/cp_data at 0.0° aoa.csv'
simulation_data = pd.read_csv(simulation_data_path)

x_simulation = simulation_data['# x']
cp_simulation = simulation_data[' Cp']"""

# Test cas
x_simulation = np.linspace(0, 100, 100)
cp_simulation = - (np.sin(np.pi * x_simulation/100))

# Create the plot
plt.figure(figsize=(10, 6))

plt.plot(x_experimental, cp_experimental, 'o', label='Experimental Data', color='blue')
plt.plot(x_simulation, cp_simulation, 'o', label='Simulation Data', color='red')

plt.gca().invert_yaxis() #inverted y axis dont foget about that!

plt.xlabel('x')
plt.ylabel('Cp')
plt.title('Pressure Coefficient vs. Chord')
plt.legend()
plt.grid(True)

plt.savefig('pressure_coefficient_comparison.png', dpi=300)
plt.show()

