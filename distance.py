import numpy as np
import types
import scipy.spatial._distance_wrap as _distance_wrap

dist_measures = {'euclidean': 'euclidean',
                 'weighted_threshold': lambda u,v: weighted_threshold(u,v),
                 'relevant_distance': lambda u,v: relevant_distance(u,v),
                 'experimental_relevant_distance' : lambda u, v: experimental_relevant_distance(u,v),
                 'sqeuclidean': 'sqeuclidean',
                 'correlation': 'correlation',
                 'cityblock': 'cityblock'}


def _copy_array_if_base_present(a):
    """
    Copies the array if its base points to a parent array.
    """
    if a.base is not None:
        return a.copy()
    elif np.issubsctype(a, np.float32):
        return np.array(a, dtype=np.double)
    else:
        return a

def _copy_arrays_if_base_present(T):
    """
    Accepts a tuple of arrays T. Copies the array T[i] if its base array
    points to an actual array. Otherwise, the reference is just copied.
    This is useful if the arrays are being passed to a C function that
    does not do proper striding.
    """
    l = [_copy_array_if_base_present(a) for a in T]
    return l

def _convert_to_bool(X):
    if X.dtype != np.bool:
        X = np.bool_(X)
    if not X.flags.contiguous:
        X = X.copy()
    return X

def _convert_to_double(X):
    if X.dtype != np.double:
        X = np.double(X)
    if not X.flags.contiguous:
        X = X.copy()
    return X



def euclidean(u, v):
    """
    Computes the Euclidean distance between two n-vectors ``u`` and ``v``,
    which is defined as

    .. math::

       {||u-v||}_2

    :Parameters:
       u : ndarray
           An :math:`n`-dimensional vector.
       v : ndarray
           An :math:`n`-dimensional vector.

    :Returns:
       d : double
           The Euclidean distance between vectors ``u`` and ``v``.
    """
    u = np.asarray(u, order='c')
    v = np.asarray(v, order='c')
    q=np.matrix(u-v)
    return np.sqrt((q*q.T).sum())

def sqeuclidean(u, v):
    """
    Computes the squared Euclidean distance between two n-vectors u and v,
    which is defined as

    .. math::

       {||u-v||}_2^2.


    :Parameters:
       u : ndarray
           An :math:`n`-dimensional vector.
       v : ndarray
           An :math:`n`-dimensional vector.

    :Returns:
       d : double
           The squared Euclidean distance between vectors ``u`` and ``v``.
    """
    u = np.asarray(u, order='c')
    v = np.asarray(v, order='c')
    return ((u-v)*(u-v).T).sum()

def correlation(u, v):
    r"""
    Computes the correlation distance between two n-vectors ``u`` and
    ``v``, which is defined as

    .. math::

       \frac{1 - (u - \bar{u}){(v - \bar{v})}^T}
            {{||(u - \bar{u})||}_2 {||(v - \bar{v})||}_2^T}

    where :math:`\bar{u}` is the mean of a vectors elements and ``n``
    is the common dimensionality of ``u`` and ``v``.

    :Parameters:
       u : ndarray
           An :math:`n`-dimensional vector.
       v : ndarray
           An :math:`n`-dimensional vector.

    :Returns:
       d : double
           The correlation distance between vectors ``u`` and ``v``.
    """
    umu = u.mean()
    vmu = v.mean()
    um = u - umu
    vm = v - vmu
    return 1.0 - (np.dot(um, vm) /
                  (np.sqrt(np.dot(um, um))
                   * np.sqrt(np.dot(vm, vm))))

def cityblock(u, v):
    r"""
    Computes the Manhattan distance between two n-vectors u and v,
    which is defined as

    .. math::

       \sum_i {(u_i-v_i)}.

    :Parameters:
       u : ndarray
           An :math:`n`-dimensional vector.
       v : ndarray
           An :math:`n`-dimensional vector.

    :Returns:
       d : double
           The City Block distance between vectors ``u`` and ``v``.
    """
    u = np.asarray(u, order='c')
    v = np.asarray(v, order='c')
    return abs(u-v).sum()

def weighted_threshold(u, v, t=0.5, w=0.25):
    u = np.asarray(u, order='c',dtype=float)
    v = np.asarray(v, order='c',dtype=float)

    return (len(u) - np.sum((u>t)&(v>t)) - np.sum((u<=t)&(v<=t))*w) / len(u)

def forbes_dist(u, v):
    u = np.asarray(u, order='c', dtype=bool)
    v = np.asarray(v, order='c', dtype=bool)

    a = (u * v).sum()
    b = (np.invert(u) * v).sum()
    c = (u * np.invert(v)).sum()
    d = (np.invert(u) * np.invert(v)).sum()
    n = a + b + c + d * 1.
    return 1 / (1 + (n * a) / ((a + b) * (a + c)))


def corr_dist(u, v):
    u = np.asarray(u, order='c', dtype=float)
    v = np.asarray(v, order='c', dtype=float)

    return (1 - u * v) / 2

def forbes_corr_dist(u_pres,u_cor,v_pres,v_cor,k):
    """
    :param u_pres: 1D bool bin presence
    :param u_cor: 1D float point-segment-correlation
    :param v_pres: 1D bool bin presence
    :param v_cor: 1D float point-segment-correlation
    :param k: number [0,1], weight between forbes and correlation
    :return: the forbes corr distance between two
    """
    return k * forbes_dist(u_pres, v_pres) + (1 - k) * corr_dist(u_cor, v_cor)

def forbes_corr_pdist(presence, corr_coeff, k):
    m = presence.shape[0]
    n = corr_coeff.shape[1] if len(corr_coeff.shape) == 2 else 1
    dm = np.zeros((n, int(m * (m - 1) / 2)), dtype=np.double)

    idx = 0
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            res = forbes_corr_dist(presence[i], corr_coeff[i], presence[j], corr_coeff[j], k)
            dm[:,idx] = res
            idx += 1

    return dm

def relevant_distance(u, v):
    u = np.asarray(u, order='c')
    v = np.asarray(v, order='c')

    return (((1 - (u+v)/2).sum() + abs(u-v).sum()) / len(u))

def experimental_relevant_distance(u, v):
    u = np.asarray(u, order='c')
    v = np.asarray(v, order='c')

    return (((1 - (u+v)/2).sum() + abs(u-v).sum()/1.5) / len(u))**3


def pdist(X, metric='euclidean'):
    r"""
    Computes the pairwise distances between m original observations in
    n-dimensional space. Returns a condensed distance matrix Y.  For
    each :math:`i` and :math:`j` (where :math:`i<j<n`), the
    metric ``dist(u=X[i], v=X[j])`` is computed and stored in the
    :math:`ij`th entry.

    :Parameters:
       X : ndarray
           An m by n array of m original observations in an
           n-dimensional space.
       metric : string or function
           The distance metric to use. The distance function can
           be 'cityblock', 'correlation', 'euclidean', 'sqeuclidean'

    :Returns:
       Y : ndarray
           A condensed distance matrix.
    """


#         21. Y = pdist(X, 'Y')
#
#           Computes the distance between all pairs of vectors in X
#           using the distance metric Y but with a more succint,
#           verifiable, but less efficient implementation.

    X = np.asarray(X, order='c')

    if X.dtype != object:
        # The C code doesn't do striding.
        [X] = _copy_arrays_if_base_present([_convert_to_double(X)])

    s = X.shape

    if len(s) != 2:
        raise ValueError('A 2-dimensional array must be passed.')

    m = s[0]
    n = s[1]
    dm = np.zeros(int(m * (m - 1) / 2), dtype=np.double)

    mtype = type(metric)
    if mtype is types.FunctionType:
        k = 0
        for i in range(0, m - 1):
            for j in range(i+1, m):
                dm[k] = metric(X[i, :], X[j, :])
                k += 1

    elif mtype is str:
        mstr = metric.lower()

        if mstr in set(['euclidean', 'euclid', 'eu', 'e']):
            #return pdist(X, euclidean)
            _distance_wrap.pdist_euclidean_wrap(_convert_to_double(X), dm)
        elif mstr in set(['sqeuclidean', 'sqe', 'sqeuclid']):
            #return pdist(X, sqeuclidean)
            _distance_wrap.pdist_euclidean_wrap(_convert_to_double(X), dm)
            dm = dm ** 2.0
        elif mstr in set(['cityblock', 'cblock', 'cb', 'c']):
            #return pdist(X, cityblock)
            _distance_wrap.pdist_city_block_wrap(X, dm)
        elif mstr in set(['correlation', 'co']):
            #return pdist(X,correlation)
            X2 = X - X.mean(1)[:,np.newaxis]
            norms = np.sqrt(np.sum(X2 * X2, axis=1))
            _distance_wrap.pdist_cosine_wrap(_convert_to_double(X2), _convert_to_double(dm), _convert_to_double(norms))
        elif mstr in dist_measures.keys():
            dist = dist_measures[mstr]
            return pdist(X,metric=dist)
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier or a function.')
    return dm


def cat_pdist(X, cat_idx, threshold=0.5, eq_under_mul=0.5):
    """
    With X containing {0,1} this performs the same as city block distance,
    but has the option to weight the sum of equal zeros in the vectors.
    This way d([1,1,1],[0,0,0]) >  d([0,0,0],[0,0,0]) > d([1,1,1],[1,1,1])
    and d([1,1,1],[1,1,1]) = 0 by default.

    !!!Seems to go towards categories with small category intervals

    :param X:
    :param cat_idx:
    :param threshold:
    :param eq_under_mul:
    :return:
    """
    cat_X = [X[::,cat] for cat in cat_idx]

    m = X.shape[0]
    n = X.shape[1]

    dm = np.zeros((len(cat_idx),m,m), dtype=float)
    dm += np.inf

    for i in range(0, m - 1):
        for j in range(i+1, m):
            for cat in range(0, len(cat_idx)):

                u = cat_X[cat][i]
                v = cat_X[cat][j]

                dm[cat][i][j] = len(u) -  np.sum((u > threshold) & (v > threshold))\
                - np.sum((u <= threshold) & (v <= threshold)) * eq_under_mul

    return dm


def extended_squareform(X, force="no", checks=True):
    """
    Wrapper method for squareform()
    used for extended distance matrices, ie. list of dms
    """

    X = [squareform(x,force,checks) for x in X]

    return np.array(X)


def squareform(X, force="no", checks=True):
    r"""
    Converts a vector-form distance vector to a square-form distance
    matrix, and vice-versa.

    :Parameters:
       X : ndarray
           Either a condensed or redundant distance matrix.

    :Returns:
       Y : ndarray
           If a condensed distance matrix is passed, a redundant
           one is returned, or if a redundant one is passed, a
           condensed distance matrix is returned.

       force : string
           As with MATLAB(TM), if force is equal to 'tovector' or
           'tomatrix', the input will be treated as a distance matrix
           or distance vector respectively.

       checks : bool
           If ``checks`` is set to ``False``, no checks will be made
           for matrix symmetry nor zero diagonals. This is useful if
           it is known that ``X - X.T1`` is small and ``diag(X)`` is
           close to zero. These values are ignored any way so they do
           not disrupt the squareform transformation.


    Calling Conventions
    -------------------

    1. v = squareform(X)

       Given a square d by d symmetric distance matrix ``X``,
       ``v=squareform(X)`` returns a :math:`d*(d-1)/2` (or
       `${n \choose 2}$`) sized vector v.

      v[{n \choose 2}-{n-i \choose 2} + (j-i-1)] is the distance
      between points i and j. If X is non-square or asymmetric, an error
      is returned.

    X = squareform(v)

      Given a d*d(-1)/2 sized v for some integer d>=2 encoding distances
      as described, X=squareform(v) returns a d by d distance matrix X. The
      X[i, j] and X[j, i] values are set to
      v[{n \choose 2}-{n-i \choose 2} + (j-u-1)] and all
      diagonal elements are zero.

    """

    X = _convert_to_double(np.asarray(X, order='c'))

    if not np.issubsctype(X, np.double):
        raise TypeError('A double array must be passed.')

    s = X.shape

    # X = squareform(v)
    if len(s) == 1 and force != 'tomatrix':
        if X.shape[0] == 0:
            return np.zeros((1,1), dtype=np.double)

        # Grab the closest value to the square root of the number
        # of elements times 2 to see if the number of elements
        # is indeed a binomial coefficient.
        d = int(np.ceil(np.sqrt(X.shape[0] * 2)))

        # Check that v is of valid dimensions.
        if d * (d - 1) / 2 != int(s[0]):
            raise ValueError('Incompatible vector size. It must be a binomial coefficient n choose 2 for some integer n >= 2.')

        # Allocate memory for the distance matrix.
        M = np.zeros((d, d), dtype=np.double)

        # Since the C code does not support striding using strides.
        # The dimensions are used instead.
        [X] = _copy_arrays_if_base_present([X])

        # Fill in the values of the distance matrix.
        _distance_wrap.to_squareform_from_vector_wrap(M, X)

        return M
    elif len(s) != 1 and force.lower() == 'tomatrix':
        raise ValueError("Forcing 'tomatrix' but input X is not a distance vector.")
    elif len(s) == 2 and force.lower() != 'tovector':
        if s[0] != s[1]:
            raise ValueError('The matrix argument must be square.')
        if checks:
            if np.sum(np.sum(X == X.T)) != np.product(X.shape):
                raise ValueError('The distance matrix array must be symmetrical.')
            if (X.diagonal() != 0).any():
                raise ValueError('The distance matrix array must have zeros along the diagonal.')

        # One-side of the dimensions is set here.
        d = s[0]

        if d <= 1:
            return np.array([], dtype=np.double)

        # Create a vector.
        v = np.zeros(((d * (d - 1) / 2),), dtype=np.double)

        # Since the C code does not support striding using strides.
        # The dimensions are used instead.
        [X] = _copy_arrays_if_base_present([X])

        # Convert the vector to squareform.
        _distance_wrap.to_vector_from_squareform_wrap(X, v)
        return v
    elif len(s) != 2 and force.lower() == 'tomatrix':
        raise ValueError("Forcing 'tomatrix' but input X is not a distance vector.")
    else:
        raise ValueError('The first argument must be one or two dimensional array. A %d-dimensional array is not permitted' % len(s))