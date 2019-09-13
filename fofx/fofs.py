import numpy as np
from numba import njit
import copy


def get_fofs(seg):
    """
    group any objects whose seg maps touch

    Parameters
    -----------
    seg: array
        Segmentationi map produced by Source Extractor

    Returns
    --------
    array with fields 'fof_id', 'fof_size' and 'number'
    """

    useg = np.unique(seg)
    w, = np.where(useg > 0)
    useg = useg[w]
    useg.sort()

    dtype = [('number', 'i4'), ('nbr_number', 'i4')]

    pairs_singleton = np.zeros(useg.size, dtype=dtype)
    pairs_singleton['number'] = useg
    pairs_singleton['nbr_number'] = useg

    pairs = np.zeros(seg.size, dtype=dtype)
    pairs['number'] = -1

    _get_seg_pairs(seg, pairs)
    w, = np.where(pairs['number'] > 0)
    pairs = pairs[w]

    pairs = np.hstack((pairs_singleton, pairs))

    pairs = _get_unique_pairs(pairs)

    nf = NbrsFoF(pairs)
    return nf.get_fofs()


class NbrsFoF(object):
    """
    extract unique groups form a list of id pairs
    """
    def __init__(self, nbrs_data):
        self.nbrs_data = nbrs_data
        self.Nobj = len(np.unique(nbrs_data['number']))

        # records fof_id of entry
        self.linked = np.zeros(self.Nobj, dtype='i8')
        self.fofs = {}

        self._fof_data = None
        self._make_fofs()

    def get_fofs(self):
        """
        get the fof groups
        """
        return self._fof_data

    def _make_fofs(self):
        """
        run the fof group finder
        """
        self._init_fofs()

        for i in range(self.Nobj):
            self._link_fof(i)

        for fof_id, k in enumerate(self.fofs):
            inds = np.array(list(self.fofs[k]), dtype=int)
            self.linked[inds[:]] = fof_id

        self.fofs = {}

        self._make_fof_data()

    def _link_fof(self, mind):
        """
        link a fof group
        """
        # get nbrs for this object
        nbrs = set(self._get_nbrs_index(mind))

        # always make a base fof
        if self.linked[mind] == -1:
            fof_id = copy.copy(mind)
            self.fofs[fof_id] = set([mind])
            self.linked[mind] = fof_id
        else:
            fof_id = copy.copy(self.linked[mind])

        # loop through nbrs
        for nbr in nbrs:
            if self.linked[nbr] == -1 or self.linked[nbr] == fof_id:
                # not linked so add to current
                self.fofs[fof_id].add(nbr)
                self.linked[nbr] = fof_id
            else:
                # join!
                self.fofs[self.linked[nbr]] |= self.fofs[fof_id]
                del self.fofs[fof_id]
                fof_id = copy.copy(self.linked[nbr])
                inds = np.array(list(self.fofs[fof_id]), dtype=int)
                self.linked[inds[:]] = fof_id

    def _make_fof_data(self):
        """
        generate the catalog from the fof sets
        """
        self._fof_data = []

        for i in range(self.Nobj):
            self._fof_data.append((self.linked[i], i+1))

        self._fof_data = np.array(
            self._fof_data,
            dtype=[('fof_id', 'i8'), ('number', 'i8')]
        )

        i = np.argsort(self._fof_data['number'])
        self._fof_data = self._fof_data[i]
        assert np.all(self._fof_data['fof_id'] >= 0)

    def _init_fofs(self):
        """
        initialize the fof groups
        """
        self.linked[:] = -1
        self.fofs = {}

    def _get_nbrs_index(self, mind):
        """
        get entries where the entered index mathes the
        first entry 'number' (not nbr_number)
        """
        q, = np.where((self.nbrs_data['number'] == mind+1)
                      & (self.nbrs_data['nbr_number'] > 0))
        if len(q) > 0:
            return list(self.nbrs_data['nbr_number'][q]-1)
        else:
            return []


@njit
def _get_seg_pairs(seg, pairs):
    """
    get pairs of ids from a seg map when neighboring pixel
    ids do not match

    Objects 1-2-3 would all be linked but 4 would not

        0 0 0 0 0 0 0 0 0 0 0 0 0
        0 1 1 1 0 0 0 0 0 0 0 0 0
        0 1 1 1 0 0 0 0 0 3 0 0 0
        0 1 1 2 2 2 0 0 3 3 3 0 0
        0 0 2 2 2 2 2 3 3 3 3 3 0
        0 2 2 2 2 0 0 0 0 3 3 0 0
        0 0 2 2 2 0 0 0 0 0 0 0 0
        0 0 0 2 0 0 0 0 4 4 0 0 0
        0 0 0 0 0 0 0 0 4 4 4 0 0
        0 0 0 0 0 0 0 0 4 4 4 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0
    """
    nrow, ncol = seg.shape

    pi = 0
    for row in range(nrow):
        rowstart = row - 1
        rowend = row + 1

        for col in range(ncol):

            ind = seg[row, col]

            if ind == 0:
                # 0 means this pixel is not assigned to an object
                continue

            colstart = col - 1
            colend = col + 1

            for crow in range(rowstart, rowend+1):
                if crow == -1 or crow == nrow:
                    continue

                for ccol in range(colstart, colend+1):
                    if ccol == -1 or ccol == ncol:
                        continue

                    if crow == row and ccol == col:
                        continue

                    cind = seg[crow, ccol]

                    if cind != 0 and cind != ind:
                        # we found a neighboring pixel assigned
                        # to another object
                        pairs['number'][pi] = ind
                        pairs['nbr_number'][pi] = cind
                        pi += 1

    npairs = pi
    return npairs


def _get_unique_pairs(pairs):
    """
    get unique pairs, assuming max seg id number is at most
    1_000_000
    """

    tid = pairs['number']*1_000_000 + pairs['nbr_number']
    uid, uid_index = np.unique(tid, return_index=True)
    return pairs[uid_index]
