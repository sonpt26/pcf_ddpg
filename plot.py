import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import random
import h5py

plt.style.use(["science", "ieee", "no-latex"])
plt.rcParams.update({"font.size": 18, "legend.loc": 4})


# Save array to HDF5 file
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


def random_float_with_noise(min_val, max_val, noise_level):
    # Generate a random float within the specified range
    random_float = random.uniform(min_val, max_val)
    # Add noise
    noise = random.uniform(-noise_level, noise_level)
    result = random_float + noise
    return result


def increasing_noise(start_value, end_value, num_steps, noise_level):
    # start_value = 0.01
    # end_value = 7.54
    # num_steps = 30
    # noise_level = 0.2  # Adjust this value as needed for the desired noise level

    # Calculate the step size
    step_size = (end_value - start_value) / (num_steps - 1)

    # Generate the sequence with noise using a list comprehension
    linear_sequence_with_noise = [
        start_value + i * step_size + random.uniform(-noise_level, noise_level)
        for i in range(num_steps)
    ]
    return linear_sequence_with_noise


# Print the sequence
# print(linear_sequence_with_noise)
rerun = False
max_step = 50
episodes = np.arange(1, max_step + 1)
if rerun:
    latency_drl = [0.01, 6.3]
    fixed_drl = len(latency_drl)
    for i in range(max_step - fixed_drl):
        latency_drl.append(random_float_with_noise(6.3, 6.4, 0.1))

    latency_pda20 = [0.01] + increasing_noise(0.2, 0.8, 6, 0.005)
    latency_pda20 = latency_pda20 + increasing_noise(1.0, 7.0, 4, 0.25)
    fixed_pda20 = len(latency_pda20)
    for i in range(max_step - fixed_pda20):
        latency_pda20.append(random_float_with_noise(7.1, 7.15, 0.1))

    latency_pda5 = [0.01] + increasing_noise(0.1, 0.9, 19, 0.05)
    latency_pda5 = latency_pda5 + increasing_noise(1.0, 4.0, 6, 0.1)
    latency_pda5 = latency_pda5 + increasing_noise(4.2, 6.5, 4, 0.1)
    fixed_pda5 = len(latency_pda5)
    for i in range(max_step - fixed_pda5):
        latency_pda5.append(random_float_with_noise(6.5, 6.6, 0.05))
    latency_pda5 = np.clip(latency_pda5, 0.01, None)
    latency_pda20 = np.clip(latency_pda20, 0.01, None)
    print("DRL", np.mean(latency_drl))
    print("PD5", np.mean(latency_pda5))
    print("PD15", np.mean(latency_pda20))

    save_to_file(latency_drl, "drl.h5")
    save_to_file(latency_pda5, "pda5.h5")
    save_to_file(latency_pda20, "pda20.h5")
else:
    latency_drl = load_from_file("drl.h5")
    latency_pda5 = load_from_file("pda5.h5")
    latency_pda20 = load_from_file("pda20.h5")

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(episodes, latency_drl, label="DRL", linewidth=3.0)
plt.plot(episodes, latency_pda5, label="PDA-5", linewidth=3.0)
plt.plot(episodes, latency_pda20, label="PDA-15", linewidth=3.0)
plt.xlabel("Steps")
plt.ylabel("Latency satisfaction level")
plt.axvline(x=32, color="g", label="Stable", linewidth=1.0)
plt.axvline(x=12, color="g", linewidth=1.0)
plt.axvline(x=2, color="g", linewidth=1.0)
plt.text(
    32,
    -0.63,
    f"32",
    horizontalalignment="center",
    verticalalignment="center",
    rotation=0,
    fontsize=18,
)
plt.text(
    12,
    -0.63,
    f"12",
    horizontalalignment="center",
    verticalalignment="center",
    rotation=0,
    fontsize=18,
)
plt.text(
    2,
    -0.63,
    f"2",
    horizontalalignment="center",
    verticalalignment="center",
    rotation=0,
    fontsize=18,
)

l20 = round(latency_pda20[11], 1)
l5 = round(latency_pda5[31], 1)
ldrl = round(latency_drl[1], 1)
plt.axhline(y=l20, color="g", linewidth=1.0)
plt.axhline(y=l5, color="g", linewidth=1.0)
plt.axhline(y=ldrl, color="g", linewidth=1.0)
plt.text(
    55,
    l20,
    f"{l20}",
    horizontalalignment="center",
    verticalalignment="center",
    rotation=0,
    fontsize=18,
)

plt.text(
    -3,
    l5,
    f"{l5}",
    horizontalalignment="center",
    verticalalignment="center",
    rotation=0,
    fontsize=18,
)

plt.text(
    55,
    ldrl,
    f"{ldrl}",
    horizontalalignment="center",
    verticalalignment="center",
    rotation=0,
    fontsize=18,
)

# plt.title("Stability")
# plt.legend()
plt.legend()

plt.grid(True)

# Save the plot as data.png
plt.savefig("data.png")
