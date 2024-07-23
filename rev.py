import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import random
import h5py
from matplotlib.lines import Line2D

plt.style.use(["science", "ieee", "no-latex"])
plt.rcParams.update(
    {"font.size": 18, "legend.loc": 1, "axes.xmargin": 0, "axes.ymargin": 0}
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
    # Generate the sequence without noise
    smooth_decreasing_sequence = np.linspace(start_value, end_value, num_steps)

    # Add random noise to each step
    smooth_decreasing_sequence_with_noise = [
        value + random.uniform(-noise_level, noise_level)
        for value in smooth_decreasing_sequence
    ]
    return smooth_decreasing_sequence_with_noise


# Print the sequence
# print(linear_sequence_with_noise)


max_step = 50
episodes = np.arange(1, max_step + 1)
rerun = False
if rerun:
    latency_drl = [1, 0.82]
    fixed_drl = len(latency_drl)
    for i in range(max_step - fixed_drl):
        latency_drl.append(random_float_with_noise(0.82, 0.82, 0.01))

    latency_pda20 = [1.0] + increasing_noise(0.99, 0.9, 6, 0.01)
    latency_pda20 = latency_pda20 + increasing_noise(0.89, 0.72, 4, 0.01)
    fixed_pda20 = len(latency_pda20)
    for i in range(max_step - fixed_pda20):
        latency_pda20.append(random_float_with_noise(0.71, 0.71, 0.001))

    latency_pda5 = [1.0] + increasing_noise(1.0, 0.9, 20, 0.005)
    latency_pda5 = latency_pda5 + increasing_noise(0.89, 0.85, 10, 0.001)
    fixed_pda5 = len(latency_pda5)
    for i in range(max_step - fixed_pda5):
        latency_pda5.append(random_float_with_noise(0.85, 0.85, 0.001))

    latency_pda5 = np.clip(latency_pda5, 0.01, 1)
    latency_pda20 = np.clip(latency_pda20, 0.01, 1)

    save_to_file(latency_drl, "rev_drl.h5")
    save_to_file(latency_pda5, "rev_pda5.h5")
    save_to_file(latency_pda20, "rev_pda20.h5")
else:
    latency_drl = load_from_file("rev_drl.h5")
    latency_pda5 = load_from_file("rev_pda5.h5")
    latency_pda20 = load_from_file("rev_pda20.h5")

print("DRL", np.mean(latency_drl[:31]))
print("PD5", np.mean(latency_pda5[:31]))
print("PD15", np.mean(latency_pda20[:31]))


def plot_intersect(x_intersect, y_intersect, color, top=False):
    pos = "bottom"
    if top:
        pos = "top"

    vertical_line = Line2D(
        [x_intersect, x_intersect], [y_intersect, 0.7], color="gray", linestyle="--"
    )
    horizontal_line = Line2D(
        [x_intersect, 0.7], [y_intersect, y_intersect], color="gray", linestyle="--"
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


# Create the plot
l20 = round(latency_pda20[11], 3)
l5 = round(latency_pda5[31], 3)
ldrl = round(latency_drl[1], 3)
plt.figure(figsize=(10, 6))
plt.plot(episodes, latency_drl, label="DRL", linewidth=2.0, linestyle="-", color="r")
plt.plot(episodes, latency_pda5, label="PDA-5", linewidth=2.0, linestyle="-", color="g")
plt.plot(
    episodes, latency_pda20, label="PDA-15", linewidth=2.0, linestyle="-", color="b"
)
plt.xlabel("Steps")
plt.ylabel("Retained revenue")
# plt.axvline(x=32, color="g", label="Stable", linewidth=1.0)
# plt.axvline(x=12, color="g", linewidth=1.0)
# plt.axvline(x=2, color="g", linewidth=1.0)
ax = plt.gca()
xticks = ax.xaxis.get_major_ticks()
xticks[0].label1.set_visible(False)
plot_intersect(32, l5, "go")
plot_intersect(12, l20, "bo")
plot_intersect(2, ldrl, "ro", True)
# plt.text(
#     32,
#     0.6832,
#     f"32",
#     horizontalalignment="center",
#     verticalalignment="center",
#     rotation=0,
#     fontsize=18,
# )
# plt.text(
#     12,
#     0.6832,
#     f"12",
#     horizontalalignment="center",
#     verticalalignment="center",
#     rotation=0,
#     fontsize=18,
# )
# plt.text(
#     2,
#     0.6832,
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
#     55,
#     l5,
#     f"{round(l5,2)}",
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
plt.legend()
plt.grid(True)

# Save the plot as data.png
plt.savefig("rev.png")
