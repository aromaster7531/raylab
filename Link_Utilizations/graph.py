import matplotlib.pyplot as plt
import numpy as np

# Function to read values from a file
def read_values(file_name):
    with open(file_name, "r") as file:
        values = [float(line.strip()) for line in file.readlines()]
    return values

# Read values from the files
baseline_values = read_values("BASELINE_averages.txt")
ppolr_values = read_values("PPOLR_averages.txt")
dqn_values = read_values("DQN_averages.txt")

# Create x-axis for each set of values based on their respective lengths
baseline_iterations = np.arange(len(baseline_values))
ppolr_iterations = np.arange(len(ppolr_values))
dqn_iterations = np.arange(len(dqn_values))

# Polynomial fitting for baseline values
baseline_poly = np.polyfit(baseline_iterations, baseline_values, deg=1)
baseline_fit = np.polyval(baseline_poly, baseline_iterations)

# Polynomial fitting for PPOLR values
ppolr_poly = np.polyfit(ppolr_iterations, ppolr_values, deg=1)
ppolr_fit = np.polyval(ppolr_poly, ppolr_iterations)

# Polynomial fitting for DQN values
dqn_poly = np.polyfit(dqn_iterations, dqn_values, deg=1)
dqn_fit = np.polyval(dqn_poly, dqn_iterations)

# Plotting Baseline
plt.figure(figsize=(10, 5))
plt.plot(baseline_iterations, baseline_fit, '-', label='Baseline Best Fit', color='red')
plt.xlabel('Iteration')
plt.ylabel('Average Value')
plt.title('Baseline Averages with Best Fit Line')
plt.legend()
plt.grid(True)
plt.show()

# Plotting PPOLR
plt.figure(figsize=(10, 5))
plt.plot(ppolr_iterations, ppolr_fit, '-', label='PPOLR Best Fit', color='green')
plt.xlabel('Iteration')
plt.ylabel('Average Value')
plt.title('PPOLR Averages with Best Fit Line')
plt.legend()
plt.grid(True)
plt.show()

# Plotting DQN
plt.figure(figsize=(10, 5))
plt.plot(dqn_iterations, dqn_fit, '-', label='DQN Best Fit', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Average Value')
plt.title('DQN Averages with Best Fit Line')
plt.legend()
plt.grid(True)
plt.show()
