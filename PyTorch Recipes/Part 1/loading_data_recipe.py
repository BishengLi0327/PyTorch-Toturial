import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

# A data point in Yesno is a tuple (waveform, sample_rate, labels) where labels
# is a list of integers with 1 for yes and 0 for no.
# yesno_data = torchaudio.datasets.YESNO(
#     './Data/YESNO',
#     download=True,
#     transform=None,
#     target_transform=None
# )
yesno_data = torchaudio.datasets.YESNO('./Data/YESNO/', download=True)

n = 3
waveform, sample_rate, labels = yesno_data[n]
print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))

data_loader = torch.utils.data.DataLoader(
    yesno_data, batch_size=1, shuffle=True
)

for data in data_loader:
    print('Data: ', data)
    print("Waveform: {}\n Sample rate: {}\n Labels: {}".format(data[0], data[1],data[2]))
    break

print(data[0][0].numpy())
plt.figure()
plt.plot(waveform.t().numpy())
plt.show()
