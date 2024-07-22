import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee", "no-latex"])
plt.rcParams.update({"font.size": 18})

# Generate a noisy signal converging to a constant value
# Generate a reward signal starting very small and gradually increasing with spikes
np.random.seed(42)
time = np.linspace(0, 200, 3000)
initial_value = 0.01
final_value = 0.82
convergence_rate = 0.04

# Create a signal that grows slowly at first and then more rapidly
slow_growth = initial_value + (final_value - initial_value) * (
    1 - np.exp(-convergence_rate * (time - 80))
) * (time > 80)
signal = np.where(time <= 80, initial_value, slow_growth)

# Add noise and spikes
base_noise = np.random.normal(0, 0.2, time.shape)
spike_noise = np.random.normal(0, 0.35, time.shape)
spikes = (
    np.random.rand(len(time)) < 0.05
) * spike_noise  # 5% chance of spike at each point
signal += base_noise + spikes

# Clip the signal to ensure it stays within the range [0.01, 1.1]
signal = np.clip(signal, 0.01, 1.1)

# Calculate the moving average
window_size = 40
moving_average = np.convolve(signal, np.ones(window_size) / window_size, mode="valid")


# Plot the original signal and the moving average
plt.figure(figsize=(10, 6))
plt.plot(time, signal, label="Reward", alpha=0.6)
plt.plot(
    time[window_size - 1 :],
    moving_average,
    label="40-Moving Average",
    color="red",
    linewidth=2,
)
# plt.axhline(final_value, color="green", linestyle="--", label="Final Value")
# plt.title("Convergence of Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.savefig("converge.png")
