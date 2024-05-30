import numpy as np
import matplotlib.pyplot as plt


data = np.load("data_log_cnn_20_True_0.001_0.01_True_True_True_True.npy")
plt.loglog(data[:, 0], 100.0 * (1.0 - data[:, 1] / 10000.0), "k", label="CNN + CNN Top")

data = np.load("data_log_cnn_20_False_0.001_0.01_True_True_True_True.npy")
plt.loglog(data[:, 0], 100.0 * (1.0 - data[:, 1] / 10000.0), "k--", label="CNN")

data = np.load("data_log_nnmf_20_True_0.001_0.01_True_True_True_True.npy")
plt.loglog(
    data[:, 0],
    100.0 * (1.0 - data[:, 1] / 10000.0),
    "r",
    label="NNMF + CNN Top (Iter 20, KL)",
)

data = np.load("data_log_nnmf_20_False_0.001_0.01_True_True_True_True.npy")
plt.loglog(
    data[:, 0],
    100.0 * (1.0 - data[:, 1] / 10000.0),
    "r--",
    label="NNMF (Iter 20, KL)",
)

data = np.load("data_log_nnmf_20_True_0.001_0.01_True_True_True_False.npy")
plt.loglog(
    data[:, 0],
    100.0 * (1.0 - data[:, 1] / 10000.0),
    "b",
    label="NNMF + CNN Top (Iter 20, MSE)",
)

data = np.load("data_log_nnmf_20_False_0.001_0.01_True_True_True_False.npy")
plt.loglog(
    data[:, 0],
    100.0 * (1.0 - data[:, 1] / 10000.0),
    "b--",
    label="NNMF (Iter 20, MSE)",
)

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Error [%]")
plt.show()
