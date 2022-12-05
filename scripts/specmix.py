import numpy as np


def specmix(spec1, spec2, label1, label2) :
    spec = spec1.copy()
    label = label1.copy()

    for i in range(spec1.shape[0]) :
        t_times = np.random.randint(2)
        f_times = np.random.randint(2)
        ratio = np.random.rand(1)[0]
        for _ in range(t_times):
            t = np.random.randint((1 - ratio) * label1.shape[1] + 1)
            spec[i, :, 5*t:5*(t + int(label1.shape[1] * ratio)), :] = spec2[i, :, 5*t:5*(t + int(label1.shape[1] * ratio)), :]
            label[i, t:t + int(label1.shape[1] * ratio)] = label2[i, t:t + int(label1.shape[1] * ratio)]

        for _ in range(f_times) :
            f = np.random.randint((1 - ratio) * spec1.shape[3] + 1)
            spec[i, :, :, f:f + int(spec.shape[3] * ratio)] = spec2[i, :, :, f:f + int(spec.shape[3] * ratio)]
            label[i] = label[i] * ratio + label2[i] * (1-ratio)

    return spec, label

def specmix_old(spec1, spec2, label1, label2) :
    spec = np.zeros(spec1.shape)
    label = np.zeros(label1.shape)
    for i in range(spec1.shape[0]) :
        mask, inv_mask, gamma = masking(spec1[0, 0], ratio = np.random.rand(1)[0])
        for j in range(spec1.shape[1]) :
            spec[i, j, :, :] = mask * spec1[i, j, :, :] + inv_mask * spec2[i, j, :, :]
        label[i] = label1[i] * gamma + label2[i] * (1-gamma)
    return spec, label


def masking(spec, ratio=np.random.rand(1)[0]) :
    mask = np.ones(spec.shape)

    t_times = np.random.randint(3)
    f_times = np.random.randint(3)

    for _ in range(t_times) :
        t = np.random.randint((1-ratio)*mask.shape[0]+1)
        mask[t:t+int(mask.shape[0]*ratio), :] = 0

    for _ in range(f_times) :
        f = np.random.randint((1-ratio)*mask.shape[1]+1)
        mask[:, f:f+int(mask.shape[1]*ratio)] = 0
    inv_mask = -1 * (mask - 1)

    gamma = mask.sum() / (mask.shape[0] * mask.shape[1])

    return mask, inv_mask, gamma