import torch
import numpy as np
import matplotlib.pyplot as plt

model = torch.load("Model_nnmf_20_True_0.001_0.01_True.pt")

w_1 = model[1].weight.cpu().detach().numpy()
b_1 = model[1].bias.cpu().detach().numpy()

w_5 = model[5].weight.cpu().detach().numpy()
b_5 = model[5].bias.cpu().detach().numpy()

w_9 = model[9].weight.cpu().detach().numpy()
b_9 = model[9].bias.cpu().detach().numpy()

w_12 = model[12].weight.cpu().detach().numpy()
b_12 = model[12].bias.cpu().detach().numpy()

plt.figure(1)
plt.subplot(2, 2, 1)
max_value = np.abs(w_1).max()
plt.imshow(w_1[:, :, 0, 0], cmap="seismic", vmin=-max_value, vmax=max_value)
plt.title(f"layer 1 -- min: {w_1.min():.2e} max: {w_1.max():.2e}")
plt.colorbar()

plt.subplot(2, 2, 2)
max_value = np.abs(w_5).max()
plt.imshow(w_5[:, :, 0, 0], cmap="seismic", vmin=-max_value, vmax=max_value)
plt.title(f"layer 5 -- min: {w_5.min():.2e} max: {w_5.max():.2e}")
plt.colorbar()

plt.subplot(2, 2, 3)
max_value = np.abs(w_9).max()
plt.imshow(w_9[:, :, 0, 0], cmap="seismic", vmin=-max_value, vmax=max_value)
plt.title(f"layer 9 -- min: {w_9.min():.2e} max: {w_9.max():.2e}")
plt.colorbar()

plt.subplot(2, 2, 4)
max_value = np.abs(w_12).max()
plt.imshow(w_12[:, :, 0, 0], cmap="seismic", vmin=-max_value, vmax=max_value)
plt.title(f"layer 12 -- min: {w_12.min():.2e} max: {w_12.max():.2e}")
plt.colorbar()
plt.show(block=False)

plt.figure(2)
plt.subplot(2, 2, 1)
plt.plot(b_1)
plt.title("layer 1 -- bias")
plt.xlabel("Neuron ID")

plt.subplot(2, 2, 2)
plt.plot(b_5)
plt.title("layer 5 -- bias")
plt.xlabel("Neuron ID")

plt.subplot(2, 2, 3)
plt.plot(b_9)
plt.title("layer 9 -- bias")
plt.xlabel("Neuron ID")

plt.subplot(2, 2, 4)
plt.plot(b_12)
plt.title("layer 12 -- bias")
plt.xlabel("Neuron ID")
plt.show()
