import numpy as np
import fofx


def get_fake_seg():
    return np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0],
        [0, 1, 1, 2, 2, 2, 0, 0, 3, 3, 3, 0, 0],
        [0, 0, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0],
        [0, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 0, 0],
        [0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 4, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])


def test_fofs_smoke(show=False):
    """
    test we can run end to end
    """

    seg = get_fake_seg()
    fofx.get_fofs(seg)

    if show:
        fofx.plot_seg(seg, show=True)


def test_fofs(show=False):
    """
    test we can run end to end
    """

    seg = get_fake_seg()
    fofs = fofx.get_fofs(seg)
    print(fofs['fof_id'])

    uids = np.unique(fofs['fof_id'])
    assert fofs.size == 4
    assert uids.size == 2

    w0, = np.where(fofs['fof_id'] == 0)
    w1, = np.where(fofs['fof_id'] == 1)

    assert w0.size == 3
    assert w1.size == 1
