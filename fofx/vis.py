import numpy as np
from numba import njit


def plot_seg(segin, width=1000, rng=None, show=False, **kw):
    """
    plot the seg map with randomized colors for better display
    """
    import images

    if rng is None:
        rng = np.random.RandomState()

    seg = np.transpose(segin)

    max_seg = seg.max()

    low = 50/255
    high = 255/255
    n = max_seg*3
    colors = rng.uniform(low=low, high=high, size=n).reshape(
        (max_seg, 3),
    )

    cseg = np.zeros((seg.shape[0], seg.shape[1], 3))

    _make_color_seg(seg, cseg, colors)

    plt = images.view(cseg, show=False, **kw)

    if show:
        srat = seg.shape[1]/seg.shape[0]
        plt.show(width=width, height=width*srat)

    return plt


@njit
def _make_color_seg(seg, cseg, colors):
    nrow, ncol = seg.shape
    for row in range(nrow):
        for col in range(ncol):

            val = seg[row, col]
            if val > 0:
                r, g, b = colors[val]
                cseg[row, col, 0] = r
                cseg[row, col, 1] = g
                cseg[row, col, 2] = b
