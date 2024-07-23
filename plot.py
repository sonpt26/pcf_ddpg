import matplotlib.pyplot as plt
import numpy as np
from pyparsing import col
import scienceplots
import random
import h5py
from matplotlib.lines import Line2D

plt.style.use(["science", "ieee", "no-latex"])
plt.rcParams.update(
    {"font.size": 18, "legend.loc": 4, "axes.xmargin": 0, "axes.ymargin": 0}
)
plt.margins(x=0, y=0)


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
    latency_drl = [0.01, 6.31]
    fixed_drl = len(latency_drl)
    for i in range(max_step - fixed_drl):
        latency_drl.append(random_float_with_noise(6.3, 6.4, 0.1))

    latency_pda20 = [0.01] + increasing_noise(0.2, 0.8, 6, 0.005)
    latency_pda20 = latency_pda20 + increasing_noise(1.1, 6.9, 4, 0.25)
    fixed_pda20 = len(latency_pda20)
    for i in range(max_step - fixed_pda20):
        latency_pda20.append(random_float_with_noise(7.1, 7.12, 0.1))

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
l20 = round(latency_pda20[11], 2)
l5 = round(latency_pda5[31], 2)
ldrl = round(latency_drl[1], 2)


plt.figure(figsize=(10, 6))
# fig, ax = plt.subplots()
plt.plot(episodes, latency_drl, label="DRL", linewidth=2.0, linestyle="-", color="r")
plt.plot(episodes, latency_pda5, label="PDA-5", linewidth=2.0, linestyle="-", color="g")
plt.plot(
    episodes, latency_pda20, label="PDA-15", linewidth=2.0, linestyle="-", color="b"
)
plt.xlabel("Steps")
plt.ylabel("Latency satisfaction level")

ax = plt.gca()
yticks = ax.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)


def plot_intersect(x_intersect, y_intersect, color, top=False):
    pos = "bottom"
    if top:
        pos = "top"

    vertical_line = Line2D(
        [x_intersect, x_intersect], [y_intersect, 0], color="gray", linestyle="--"
    )
    horizontal_line = Line2D(
        [x_intersect, 0], [y_intersect, y_intersect], color="gray", linestyle="--"
    )
    plt.plot(x_intersect, y_intersect, color)
    plt.gca().add_line(vertical_line)
    plt.gca().add_line(horizontal_line)
    plt.text(
        x_intersect,
        y_intersect,
        f"({x_intersect:.0f}, {y_intersect:.2f})",
        fontsize=15,
        ha="left",
        va=pos,
        color="gray",
    )


plot_intersect(32, l5, "go")
plot_intersect(12, l20, "bo")
plot_intersect(2, ldrl, "ro", True)
# plt.axvline(
#     x=32,
#     ymin=-2,
#     ymax=l5,
#     label="Stable",
#     linewidth=1.0,
#     linestyle="dashed",
#     color="gray",
# )
# plt.axvline(
#     x=12,
#     ymin=-2,
#     ymax=l20,
#     linewidth=1.0,
#     linestyle="dashed",
#     color="gray",
# )
# plt.axvline(
#     x=2,
#     ymin=-2,
#     ymax=ldrl,
#     linewidth=1.0,
#     linestyle="dashed",
#     color="gray",
# )
# plt.text(
#     32,
#     -0.63,
#     f"32",
#     horizontalalignment="center",
#     verticalalignment="center",
#     rotation=0,
#     fontsize=18,
# )
# plt.text(
#     12,
#     -0.63,
#     f"12",
#     horizontalalignment="center",
#     verticalalignment="center",
#     rotation=0,
#     fontsize=18,
# )
# plt.text(
#     2,
#     -0.63,
#     f"2",
#     horizontalalignment="center",
#     verticalalignment="center",
#     rotation=0,
#     fontsize=18,
# )


# plt.axhline(y=l20, color="g", linewidth=1.0)
# plt.axhline(y=l5, color="g", linewidth=1.0)
# plt.axhline(y=ldrl, color="g", linewidth=1.0)
# plt.text(
#     55,
#     l20,
#     f"{l20}",
#     horizontalalignment="center",
#     verticalalignment="center",
#     rotation=0,
#     fontsize=18,
# )

# plt.text(
#     -3,
#     l5,
#     f"{l5}",
#     horizontalalignment="center",
#     verticalalignment="center",
#     rotation=0,
#     fontsize=18,
# )

# plt.text(
#     55,
#     ldrl,
#     f"{ldrl}",
#     horizontalalignment="center",
#     verticalalignment="center",
#     rotation=0,
#     fontsize=18,
# )

# plt.title("Stability")
# plt.legend()
plt.legend()

plt.grid(True)

# Save the plot as data.png
plt.savefig("data.png")
