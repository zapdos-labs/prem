import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

# -----------------------------
# 1. Generate a piecewise constant signal
# -----------------------------
n, dim = 50, 3       # number of samples, dimension
n_bkps_true, sigma = 4, 5  # true number of change points, noise std

signal, bkps_true = rpt.pw_constant(n, dim, n_bkps_true, noise_std=sigma)
print(signal.shape)
# -----------------------------
# 2. Change point detection without knowing number of segments
# -----------------------------
model = "l2"
algo = rpt.Window(width=10, model=model).fit(signal)

# Use penalty to automatically decide number of change points
penalty = np.log(n) * dim * sigma**2
my_bkps = algo.predict(pen=penalty)

# -----------------------------
# 3. Display results
# -----------------------------
rpt.show.display(signal, bkps_true, my_bkps, figsize=(10, 6))
plt.savefig("output.png")

# Optional: print detected breakpoints
print("Detected breakpoints:", my_bkps)
print("True breakpoints:", bkps_true)
