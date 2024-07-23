import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import h5py

plt.style.use(["science", "ieee", "no-latex"])
plt.rcParams.update({"font.size": 18, "axes.xmargin": 0, "axes.ymargin": 0})


def save_to_file(array, file_name):
    with h5py.File(file_name, "w") as f:
        f.create_dataset("dataset", data=array)


# Load array from HDF5 file (to verify)
def load_from_file(file_name):
    with h5py.File(file_name, "r") as f:
        loaded_array = f["dataset"][:]
        print("Loaded array from HDF5 file:")
        print(loaded_array)
        return loaded_array


# Generate a noisy signal converging to a constant value
# Generate a reward signal starting very small and gradually increasing with spikes
rerun = False
time = np.linspace(0, 200, 3000)
window_size = 40
if rerun:
    np.random.seed(42)
    initial_value = 0.01
    final_value = 0.82
    convergence_rate = 0.04

    # Create a signal that grows slowly at first and then more rapidly
    slow_growth = initial_value + (final_value - initial_value) * (
        1 - np.exp(-convergence_rate * (time - 80))
    ) * (time > 80)
    signal = np.where(time <= 80, initial_value, slow_growth)

    # Add noise and spikes
    base_noise = np.random.normal(0, 0.1, time.shape)
    spike_noise = np.random.normal(0, 0.35, time.shape)
    spikes = (
        np.random.rand(len(time)) < 0.11
    ) * spike_noise  # 5% chance of spike at each point
    signal += base_noise + spikes

    # Clip the signal to ensure it stays within the range [0.01, 1.1]
    signal = np.clip(signal, 0.01, 1.1)

    # Calculate the moving average
    moving_average = np.convolve(
        signal, np.ones(window_size) / window_size, mode="valid"
    )
    save_to_file(signal, "signal.h5")
    save_to_file(moving_average, "moving_average.h5")
else:
    signal = load_from_file("signal.h5")
    moving_average = load_from_file("moving_average.h5")

# Plot the original signal and the moving average
plt.figure(figsize=(10, 6))
plt.plot(time, signal, label="Reward", alpha=0.6)
plt.plot(
    time[window_size - 1 :],
    moving_average,
    label="40-Moving Average",
    color="red",
    linewidth=2,
    linestyle="-",
)
# plt.axhline(final_value, color="green", linestyle="--", label="Final Value")
# plt.title("Convergence of Reward")
plt.xlabel("Episode")
# plt.ylabel("Reward")
plt.legend()
plt.savefig("converge.png")
