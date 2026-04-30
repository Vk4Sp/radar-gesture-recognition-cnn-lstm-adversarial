import matplotlib.pyplot as plt
import numpy as np
from data_loader import SoliDataset

def get_sample_by_label(dataset, label):
    for i in range(len(dataset)):
        X, y = dataset[i]
        if int(y) == label:
            return X, y
    return None, None

# -------- Config --------
DATA_PATH = r"C:\Users\Venkatesan T\PycharmProjects\PythonProject1\project\dsp"
BATCH_SIZE = 8

dataset = SoliDataset(DATA_PATH, allowed_sessions=[2])  # pick one session

target_labels = [0, 3, 5]  # pick any 3 gestures

samples = []

for label in target_labels:
    X, y = get_sample_by_label(dataset, label)
    if X is not None:
        samples.append((X, y))

print("Shape:", X.shape)
print("Label:", y)

X = X.squeeze(1)  # now (40, 32, 32)

for idx, (X, y) in enumerate(samples):

    print("Gesture:", int(y))

    X = X.squeeze(1)

    # ---- Single frame ----
    plt.imshow(X[10], cmap='viridis')
    plt.title(f"Gesture {int(y)} - Frame t=10")
    plt.colorbar()
    plt.show()

    # ---- Multiple frames ----
    fig, axes = plt.subplots(1, 4, figsize=(12,3))
    times = [0, 10, 20, 30]

    for i, t in enumerate(times):
        axes[i].imshow(X[t], cmap='viridis')
        axes[i].set_title(f"t={t}")
        axes[i].axis('off')

    plt.suptitle(f"Gesture {int(y)} - Frames")
    plt.show()

    # ---- Doppler vs Time ----
    X_np = X.numpy()
    doppler_time = np.sum(X_np, axis=1)

    plt.imshow(doppler_time, aspect='auto', cmap='jet')
    plt.title(f"Gesture {int(y)} - Doppler vs Time")
    plt.xlabel("Doppler")
    plt.ylabel("Time")
    plt.colorbar()
    plt.show()

    # ---- Range vs Time ----
    range_time = np.sum(X_np, axis=2)

    plt.imshow(range_time, aspect='auto', cmap='jet')
    plt.title(f"Gesture {int(y)} - Range vs Time")
    plt.xlabel("Range")
    plt.ylabel("Time")
    plt.colorbar()
    plt.show()