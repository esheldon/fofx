import numpy as np


def plot_seg(segin, width=1000, rng=None, show=False, **kw):
    """
    plot the seg map with randomized colors for better display
    """
    import images

    if rng is None:
        rng = np.random.RandomState()

    seg = np.transpose(segin)

    cseg = np.zeros((seg.shape[0], seg.shape[1], 3))

    useg = np.unique(seg)[1:]

    low = 50/255
    high = 255/255

    for i, segval in enumerate(useg):

        w = np.where(seg == segval)

        r, g, b = rng.uniform(low=low, high=high, size=3)

        cseg[w[0], w[1], 0] = r
        cseg[w[0], w[1], 1] = g
        cseg[w[0], w[1], 2] = b

    plt = images.view(cseg, show=False, **kw)

    if show:
        srat = seg.shape[1]/seg.shape[0]
        plt.show(width=width, height=width*srat)

    return plt
