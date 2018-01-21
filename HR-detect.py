import numpy as np
from matplotlib import pyplot as plt
import peakutils
from pylibneteera.signal_processing.utils import fourier_transform, peaks
from pylibneteera.utils.plotters import plot_gt, plot_peaks
from pylibneteera.utils.utils import break_into_windows
from pylibneteera.datasets.loader import get_session_by_id


def ground_truth_hr(data=None, sample_freq=1):
    if data[0].dtype.kind not in np.typecodes['AllInteger']:
        raise ValueError
    av_instantaneous_hr = np.mean(np.ediff1d(np.argwhere(peaks(np.squeeze(data))))) / sample_freq
    return 60 / av_instantaneous_hr


def distance_from_nearest(array, value):
    return min(np.abs(array - value))


root_dir = r"G:\Team Drives\Neteera Virtual Server\Data"

x, y = get_session_by_id(26, data_loc=root_dir)

window_length = 2 ** 13

fs = 290

x = break_into_windows(x, win_len=window_length, phase=0)

(freq_r, ft) = fourier_transform(x[8, :, 13], n=len(x[8, :, 13]), fs=fs)

freq_r = freq_r[10:]
ft = ft[10:]

pks = peakutils.indexes(np.asarray(ft), thres=.02, min_dist=3)

gt = 0.
channel_deltas = []
for c in range(16):
    deltas = []
    for p in range(1000, 1800):
        delta = 0
        y1 = break_into_windows(y, win_len=window_length, phase=p)
        for i, w in enumerate(y1):
            gt = ground_truth_hr(w, sample_freq=fs)
            (freq_r1, ft1) = fourier_transform(x[i, :, c], n=2 * len(x[i, :, c]), fs=fs)
            freq_r1 = freq_r1[10:]
            ft1 = ft1[10:]
            pks1 = peakutils.indexes(np.asarray(ft1), thres=.02, min_dist=3)
            for j in range(3):
                delta += min(np.abs(freq_r1[pks1] - (j + 1) * gt))
        if delta > 1.:
            deltas.append(delta)
        else:
            deltas.append(300.)
        print("CHANNEL: ", c, "   LAG: ", p, "   DELTA: ", delta)
    true_lag = np.argmin(np.asarray(deltas))
    channel_deltas.append(1000 + true_lag)
    print("Channel: ", c, "     MIN DELTA: ", 1000 + np.argmin(np.asarray(deltas)))

print("Channel deltas: ", channel_deltas)

np.save("time-sync", channel_deltas)

y = break_into_windows(y, win_len=window_length, phase=1380)

plt.plot(freq_r, ft)

plot_peaks(freq_r, ft, pks)

for i in range(3):
    plot_gt(ft, gt=(i + 1) * gt)

plt.show()
