import numpy as np
import distance
import time
import sys
import math
from collections import defaultdict

linkage_methods = {'single': lambda dik, djk: np.min([dik, djk], axis=0),
                   'complete': lambda dik, djk: np.max([dik, djk], axis=0),
                   'average': lambda dik, djk, si, sj: (si / (si + sj)) * dik + (sj / (sj + si)) * djk,
                   'median': lambda dik, djk, dij: dik / 2 + djk / 2 - dij / 4,
                   'centroid': lambda dik, djk, dij, si, sj: (si / (si + sj)) * dik + (sj / (si + sj)) * djk - ((
                                                                                                                si * sj) / (
                                                                                                                si + sj) ** 2) * dij,
                   'medium variance': lambda dik, djk, dij, si, sj, sk: ((si + sk) / (si + sj + sk)) * dik + ((
                                                                                                              sj + sk) / (
                                                                                                              si + sj + sk)) * djk - (
                                                                                                                                     sk / (
                                                                                                                                     si + sj + sk)) * dij}


def stdout_print(*args):
    for arg in args:
        sys.stdout.write("".join(map(str, arg)) if type(arg) == tuple else "%s" % arg)
    sys.stdout.write("\n")


write = stdout_print
result_dir = "results/"


# Functions to help speed up code / debug
def id(x):
    """
    :param x: numpy array
    :return: memory block address of the array
    """

    return x.__array_interface__['data'][0]


def get_data_base(arr):
    """
    :param arr: numpy array
    :return: the base array that "owns" the actual data in memory
    """

    base = arr
    while isinstance(base.base, np.ndarray):
        base = base.base
    return base


def arrays_share_data(x, y):
    """
    :param x: numpy array
    :param y: numpy array
    :return: boolean - if they share the same data in memory
    """
    return get_data_base(x) is get_data_base(y)


def row_idx(k, n):
    """
    :param k: condensed index
    :param n: length of squared matrix
    :return: row index of squared matrix
    """
    return int(math.ceil((1 / 2.) * (- (-8 * k + 4 * n ** 2 - 4 * n - 7) ** 0.5 + 2 * n - 1) - 1))


def elem_in_i_rows(i, n):
    """
    :param i: row index (in practise row index + 1)
    :param n: length of squared matrix
    :return: position in row
    """
    return int(i * (n - 1 - i) + (i * (i + 1)) / 2)


def col_idx(k, i, n):
    """
    :param k: condensed index
    :param i: row index of squared matrix
    :param n: length of squared matrix
    :return: column index of squared matrix
    """
    return int(n - elem_in_i_rows(i + 1, n) + k)


def condensed_to_square(k, n):
    """
    Converts from condensed index to square indexes

    :param k: condensed index
    :param n: length of squared matrix
    :return: tuple(i,j) - the indices of the squared matrix
    """

    i = row_idx(k, n)
    j = col_idx(k, i, n)
    return i, j


def square_to_condensed(i, j, n):
    """
    Convects square indexes to condensed index

    :param i: index of squared matrix
    :param j: index of squared matrix
    :param n: length of squared matrix
    :return: condensed index
    """
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return int(n * j - j * (j + 1) / 2 + i - 1 - j)


def idx_to_vec(i, N):
    """
    :param i: squared index i
    :param N: number of total elements
    :return: vector for index i
    """
    return np.array([square_to_condensed(i, x, N) for x in range(N) if x != i])


def idxs_to_vecs(i, j, N):
    """
    Calculates and returns all the indexes in the dense distance matrix
    where i is a member j is a member, but excludes where both are members.
    Spacial case of idx_to_vec(i, n)

    :param i: squared index i
    :param j: squared index j
    :param N: number of total elements
    :return: vector for index i, vector for index j
            (both as numpy array for 70-80% performance boost when using for indexing)
    """
    vec_a = []
    vec_b = []

    for x in range(N):
        if x != i and x != j:
            vec_a.append(square_to_condensed(i, x, N))
            vec_b.append(square_to_condensed(j, x, N))

    return np.array(vec_a), np.array(vec_b)


class Tree:
    """
    A tree representation of clusters
    """

    @staticmethod
    def Z_to_tree_dict(Z, cats):
        sub_trees = {}
        N = Z.shape[0] + 1

        for i in range(N):
            sub_trees[i] = Tree(id=i)

        for i in range(0, Z.shape[0]):
            left = sub_trees[Z[i][0]]
            #sub_trees.pop(Z[i][0])
            right = sub_trees[Z[i][1]]
            #sub_trees.pop(Z[i][1])
            dist = Z[i][2]

            if i < len(cats):
                for c in cats[i]:
                    if type(c) != np.int64:
                        print(cats[i],2)
                tree = Tree(N + i, dist=dist, left=left, right=right, cat=cats[i])
            else:
                tree = Tree(N + i, dist=dist, left=left, right=right, cat=[])
            sub_trees[i+N] = tree
        return sub_trees



    def __init__(self, id, step=-1, dist=0, left=None, right=None, cat=[]):
        self.step = step
        self.dist = dist
        self.cat = cat
        self.left = left
        self.right = right
        self.id = id

        if left != None and right != None:
            self.leaves = left.leaves + right.leaves
        else:
            self.leaves = [id]

    def __repr__(self):
        """
        [left ( cat ) right] for each node if not leaf


        """

        if self.left is None and self.right is None:
            return str(self.id)

        # Quick fix for too large trees
        try:
            res = "[%s (%s) %s]" % (self.left.__repr__(), self.cat, self.right.__repr__())
        except RuntimeError:
            res = "[ * (%s) * ]" % self.cat

        return res

    def get_cat_counts(self):
        counts = defaultdict(int)
        stack = [self]

        while stack:
            v = stack.pop()
            for c in v.cat:
                counts[c] += 1

            if v.left is not None:
                stack.append(v.left)
            if v.right is not None:
                stack.append(v.right)

        return dict(counts)

    def get_cat_counts_as_arr(self, max_val):
        count_dict = self.get_cat_counts()
        counts = np.zeros(max_val)

        for key in count_dict.keys():
            counts[key] = count_dict[key]

        return counts

    def get_sub_cats(self):
        cat_set = set()
        stack = [self]

        while stack:
            v = stack.pop()

            for cat in v.cat:
                cat_set.add(cat)

            if v.left is not None:
                stack.append(v.left)
            if v.right is not None:
                stack.append(v.right)

        return cat_set

    def get_leaf_ids(self):
        ids = []
        stack = [self]

        while stack:
            v = stack.pop()

            if v.left is not None:
                stack.append(v.left)
            if v.right is not None:
                stack.append(v.right)

            if v.left is None and v.right is None:
                ids.append(v.id)

        return ids

    def get_ids(self):
        ids = []
        stack = [self]

        while stack:
            v = stack.pop()

            if v.left is not None:
                stack.append(v.left)
            if v.right is not None:
                stack.append(v.right)

            ids.append(v.id)

        return ids

    def get_ids_over_cutoff(self, cutoff, ids):

        if self.left is not None:
            ids.append(self.left.id)
        if self.right is not None:
            ids.append(self.right.id)

        if cutoff > 0:
            if self.left is not None:
                self.left.get_ids_at_cutoff(cutoff - 1, ids)
            if self.right is not None:
                self.right.get_ids_at_cutoff(cutoff - 1, ids)

        return ids

    def get_leaves_from_cluster_ids(self, cluster_ids):

        id_leaf_map = {}
        stack = [self]

        while stack:
            v = stack.pop()

            if v.left is not None:
                stack.append(v.left)
            if v.right is not None:
                stack.append(v.right)

            if v.id in cluster_ids:
                id_leaf_map[v.id] = v.get_leaf_ids()

        return id_leaf_map



def get_min_cat_dist(dM, idx):
    cat = np.argmin(dM[::, idx])
    dist = dM[cat, idx]

    return cat, dist


def get_min(dM):
    """
    Finds the index of the the lowest value in the extended distance matrix
    that corresponds to dM[category][squared_to_condensed(vector_a, vector_b)]

    :param dM: matrix of distance matrices
    :return: condensed index, category, distance

    >>> get_min(np.array([[0]]))
    (0, 0, 0)

    >>> get_min(np.array([[5,2,1],[3,1,2],[0,3,4],[4,19,0]]))
    (0, 2, 0)

    >>> get_min(np.array([[5,2,1,4,3,1,1,2,1,2,3,4,4,1,0],[5,2,1,4,3,1,1,2,1,2,3,4,4,1,19]]))
    (14, 0, 0)
    """

    cat, k = np.unravel_index(np.argmin(dM), dM.shape)
    min_dist = dM[cat, k]

    return k, cat, min_dist


def get_mins(dM, weights, N):
    """
    Finds the indices of the the lowest value in the 2D matrix
    that corresponds to dms[category][square_to_condensed(vector_a, vector_b)]

    :param N:
    :param weights:
    :param dM: matrix of distance matrices
    :return: index of vector a, vector b and category
    """
    # Max number of distances to get
    n = weights[-1].shape[0]

    # Storing distances
    dist = np.zeros(n, dtype=float)
    dist += np.Inf

    # Storing categories
    cat = np.zeros(n, dtype=int)
    cat -= 1

    # Mask for relevant categories
    mask = np.zeros(dM.shape[0], dtype=bool)
    mask[:] = True

    idx, cat[0], dist[0] = get_min(dM)
    mask[cat[0]] = False
    for i in range(1, n):
        new_cat, dist[i] = get_min_cat_dist(np.compress(mask, dM, axis=0), idx)
        cat[i] = new_cat

        # Break if lowest distance found was infinity
        if dist[i] == np.Inf:
            break
        cat[i] += (mask[:cat[i-1] + 1] == False).sum()
        mask[cat[i]] = False


    cat = cat[:np.sum(dist != np.Inf)]
    dist = dist[:np.sum(dist != np.Inf)]

    vec_a, vec_b = condensed_to_square(idx, N)

    dist *= weights[len(dist) - 1]
    return vec_a, vec_b, cat, dist[np.invert(np.isnan(dist))].sum()


def get_mins_wo_weights(dM, N, num_cats=3):
    """
        Finds the indices of the the lowest value in the 2D matrix
        that corresponds to dms[category][square_to_condensed(vector_a, vector_b)]

        :param N:
        :param dM: matrix of distance matrices
        :return: index of vector a, vector b and category
        """
    cats = []

    # Mask for relevant categories
    mask = np.zeros(dM.shape[0], dtype=bool)
    mask[:] = True

    idx, cat, dist = get_min(dM)
    cats.append(cat)
    for i in range(1, num_cats):
        mask[cats[-1]] = False
        new_cat, new_dist = get_min_cat_dist(np.compress(mask, dM, axis=0), idx)
        # Break if lowest distance found was infinity
        if new_dist == np.Inf:
            break
        cats.append(new_cat)
        cats[-1] += np.sum(np.invert(mask[:cats[-2] + 1]))

    vec_a, vec_b = condensed_to_square(idx, N)

    return vec_a, vec_b, np.array(cats), dist


def get_weights(lock_to, weight_scale, weight_list=[]):
    """
    :param lock_to: number of categories to lock to, i.e. the number of weights
    :param weight_scale: the next in the list
    :param weight_list
    :return:
    """

    assert lock_to >= 1, "Weights can not exist without locking (lock_to: %d)" % lock_to

    if weight_scale == 0:
        if lock_to == 1:
            weight_list.append(np.array([1.]))
            return weight_list

        get_weights(lock_to - 1, weight_scale, weight_list)
        weight_list.append(np.zeros(lock_to) + 1. / lock_to)

        return weight_list

    if lock_to == 1:
        weight_list.append(np.array([1.]))
        return weight_list

    if weight_scale < .5:
        weight_scale = 1 - weight_scale

    weights = np.zeros(lock_to)
    weights[0] = weight_scale
    for i in range(1, weights.shape[0] - 1):
        weights[i] = (1 - np.sum(weights[:i])) * weight_scale

    weights[-1] = 1 - np.sum(weights[:-1])

    get_weights(lock_to - 1, weight_scale)
    weight_list.append(weights)

    return weight_list


def linkage(X, cat_idx, method='single', metric='euclidean', lock_to=3, weight_scale=1., dM=None, write_linkage=False,
            join=False):
    """
    :param X:
        (#vectors, #features)-shaped 2D array to do clustering on
    :param cat_idx:
        (#categories, * )-shaped 2D array containing integers describing
        which indices (features or parts of VS) that belongs to which category
    :param method: str
        linkage method (complete, single, average, ...)
    :param metric: str or function
         distance measure to construct distance matrix

    :return:
        Z: the hierarchical clustering encoded as a linkage matrix
    """

    assert method in linkage_methods.keys(), "Linkage method not supported."

    if dM is None:
        write("constructing extended distance matrix...")

        # Condensed distance matrix (dm)
        # Only use features for specific category when creating dm for that category
        dM = np.array([distance.pdist(X[::, idx], metric) for idx in cat_idx])

        write("starting clustering...")
    else:
        # If dM is on squareform, convert to dense
        if len(dM.shape) == 3:
            dM = distance.squareform(dM)
        else:
            assert len(dM.shape) == 2, "Extended distance matrix needs to have shape (2,N)"

    N = int(math.ceil(math.sqrt(len(dM[0]) * 2)))

    assert len(cat_idx) >= lock_to, "Can't lock to more categories than there exists (cats: %i, lock_to: %d)" % (
    len(cat_idx), lock_to)

    # linkage matrix - scipy convention to also include span of new cluster
    Z = np.zeros((N - 1, 4), dtype=np.double)

    # Simple representation of clusters as tree objects, one for each root
    # to supply the cluster matrix with information and easier debugging
    trees = [Tree(id=i) for i in range(N)]

    if lock_to > 0:
        # Necessary for the algorithm to combine more then one distance when using several categories
        weights = np.array(get_weights(lock_to=lock_to, weight_scale=weight_scale))

    write("Clustering %d objects with %d categories" % (N, len(cat_idx)))

    step = 0
    clustered_cats = []

    # While there is more distances != np.inf in the distance matrix
    while gotValue(dM):

        # If the algorithms locks to categories or not
        if lock_to == 0:
            k, used_cat, min_dist = get_min(dM)
            x, y = condensed_to_square(k, N)
            cat_set = trees[x].get_sub_cats().union(trees[y].get_sub_cats())
            cat_set.add(used_cat)
            cats = list(cat_set)

            vec_a, vec_b = idxs_to_vecs(x, y, N)
            dM = fix_dm(dM, vec_a, vec_b, min_dist, used_cat, len(trees[x].leaves) * 1., len(trees[y].leaves) * 1.,
                        method)
            dM[::, idx_to_vec(y, N)] = np.inf
            clustered_cats.append(used_cat)
        else:
            x, y, cats, min_dist = get_mins_wo_weights(dM=dM, N=N, num_cats=lock_to)
            #x, y, cats, min_dist = get_mins(dM, weights, N)

            # to use the join of parent clusters categories as new categories
            if join:
                cats = list(trees[x].get_sub_cats().union(trees[y].get_sub_cats()).union(cats))

            dM = fix_dms(dM, x, y, min_dist, cats, len(trees[x].leaves) * 1., len(trees[y].leaves) * 1., N, method)
            clustered_cats.append(cats)

        tree_xy = Tree(N + step, step, min_dist, trees[x], trees[y], cats)

        Z[step][0] = trees[x].id
        Z[step][1] = trees[y].id
        Z[step][2] = min_dist
        Z[step][3] = len(tree_xy.leaves)

        trees[x] = tree_xy
        trees[y] = None

        step += 1

        if (step % (N//10) == 0 if N > 10 else True):
            inf_cov = (np.sum(dM == np.inf) * 1. / (dM.shape[0] * dM.shape[1]))
            write("\nStep: %d of [%d-%d]" % (step, N//2, N))
            write("Inf coverage: %.2f" % (inf_cov*100))


    trees = [tree for tree in trees if tree is not None]
    Z = connect_clusters(Z, trees, step)
    trees = [tree for tree in trees if tree is not None]

    if write_linkage:
        linkage_fn = "linkage_matrix_%s_%s_%s.link" % (N, len(cat_idx), lock_to)
        write("Writing linkage matrix to disk: %s" % linkage_fn)
        np.save(linkage_fn, Z)

    return Z, clustered_cats, trees

def fix_dm(dM, vec_a, vec_b, dij, cat, si, sj, method):
    """
    :param dM: 3D np.array with distance matrix for each category
    :param cat: category of the latest clustering
    :param i: the index of the row and column containing distances
              from cluster i (one of the old clusters in the latest clustering)
              to all the other clusters
    :param j: the index of the row and column containing distances
              from cluster i (one of the old clusters in the latest clustering)
              to all the other clusters
    :param si: the size of cluster i
    :param sj: the size of cluster j
    :param method: linkage method
    :return: modified/corrected dM with np.inf where clusters can't
             be formed, and recalculated distances for new cluster
             in the lowest index of sub clusters of the new cluster.


    >>> dM = np.array([[2.,0.,4.,1.,5.,4.]])
    >>> vec_a,vec_b = idxs_to_vecs(0,1,4)
    >>> dij = dM[0,square_to_condensed(0,1,4)]
    >>> fix_dm(dM,vec_a,vec_b,dij,0,2.,1.,'complete')
    array([[ 2.,  1.,  5.,  1.,  5.,  4.]])

    >>> dM = np.array([[[0.,5.,3.,4.],\
                        [5.,0.,7.,4.],\
                        [3.,7.,0.,4.],\
                        [4.,4.,4.,0.]]])
    >>> dM = distance.extended_squareform(dM)
    >>> vec_a,vec_b = idxs_to_vecs(0,2,4)
    >>> dij = dM[0,square_to_condensed(0,2,4)]
    >>> distance.extended_squareform(fix_dm(dM,vec_a,vec_b,dij,0,1.,1.,'average'))
    array([[[ 0.,  6.,  3.,  4.],
            [ 6.,  0.,  7.,  4.],
            [ 3.,  7.,  0.,  4.],
            [ 4.,  4.,  4.,  0.]]])

    >>> dM = np.array([[[0.,5.,3.,4.],\
                        [5.,0.,7.,4.],\
                        [3.,7.,0.,4.],\
                        [4.,4.,4.,0.]]])
    >>> dM = distance.extended_squareform(dM)
    >>> vec_a,vec_b = idxs_to_vecs(0,2,4)
    >>> dij = dM[0,square_to_condensed(0,2,4)]
    >>> distance.extended_squareform(fix_dm(dM,vec_a,vec_b,dij,0,1.,1.,'single'))
    array([[[ 0.,  5.,  3.,  4.],
            [ 5.,  0.,  7.,  4.],
            [ 3.,  7.,  0.,  4.],
            [ 4.,  4.,  4.,  0.]]])
    """

    # Get indices were neither vector positions are equal to inf
    relevant = (dM[cat][vec_a] != np.Inf) * (dM[cat][vec_b] != np.Inf)
    vec_a = np.compress(relevant, vec_a)
    vec_b = np.compress(relevant, vec_b)

    l = linkage_methods[method]

    dik = dM[cat][vec_a]
    djk = dM[cat][vec_b]

    if method == 'single':
        linked = l(dik, djk)
    elif method == 'complete':
        linked = l(dik, djk)
    elif method == 'average':
        linked = l(dik, djk, si, sj)
    elif method == 'median':
        linked = l(dik, djk, dij)
    elif method == 'centroid':
        linked = l(dik, djk, dij, si, sj)
    elif method == 'medium variance':
        raise NotImplemented("Need a way to get #leaves in clusters (sk)")
        # linked = l(dik,djk,dij,si,sj,sk)
    else:
        raise Exception

    # Set distances from new cluster ij to k in previous i-indexes
    dM[cat][vec_a] = linked

    return dM


def fix_dms(dM, i, j, dij, cats, si, sj, N, method):
    vec_a, vec_b = idxs_to_vecs(i, j, N)

    for cat in cats:
        dM = fix_dm(dM, vec_a, vec_b, dij, cat, si, sj, method)

    # Remove distances from previous j cluster indexes
    dM[::, idx_to_vec(j, N)] = np.inf

    # Make locked out categories unreachable
    for cat in range(dM.shape[0]):
        if cat not in cats:
            dM[cat][vec_a] = np.inf

    return dM


def gotValue(dM):
    for i in range(dM.shape[0]):
        for j in range(dM.shape[1]):
            if dM[i][j] != np.Inf:
                return True

    return False


def connect_clusters(Z, trees, step):
    final_dists = np.nan

    while step < len(Z):
        for x in range(len(trees)):
            if trees[x] != None:
                break

        for y in range(len(trees) - 1, 0, -1):
            if trees[y] != None:
                break

        tree_x = trees[x]
        tree_y = trees[y]
        trees[y] = None
        tree_xy = Tree(len(Z) + 1 + step, step, final_dists, tree_x, tree_y, -1)
        trees[x] = tree_xy

        Z[step][0] = tree_x.id
        Z[step][1] = tree_y.id
        Z[step][2] = final_dists
        Z[step][3] = len(tree_xy.leaves)

        step += 1

    return Z


def test_with_random_values(n=100, m=50, c=8, lock_to=0, method='complete', time_it=True):
    # n = tracks
    # m = ssms
    # c = clusters

    assert m > c
    # import scipy

    X = np.random.rand(n, m)

    start_range = range(0, m if m % c == 0 else m - m / c, m / c)
    end_range = range(m / c, m, m / c)
    if len(end_range) < len(start_range):
        end_range.append(m)

    cats = [range(x, y) for x, y in zip(start_range, end_range)]
    cats[-1] = range((m / c) * (c - 1), m)

    t = time.time()
    write("Testing with n=%d, m=%d and c=%d" % (n, m, c))

    Z = linkage(X, cats, lock_to=lock_to, method=method, weight_scale=1., write_linkage=False)

    if time_it:
        write("Time: ", time.time() - t)

    dists = Z[::, 2].copy()
    dists = np.sort(dists)
    write(dists == Z[::, 2])
    write((dists != Z[::, 2]).sum() - np.isnan(Z[::, 2]).sum())

    # hi.dendrogram(Z)
    # plt.show()
    return (dists != Z[::, 2]).sum() - np.isnan(Z[::, 2]).sum() == 0


def test_with_different_parameters():
    n = 4
    c = 5
    while test_with_random_values(n, c=c, lock_to=5, method='complete', time_it=True):
        n += 1
        if n == 73:
            n = 3
            c += 1
            if c == 60:
                c = 5


def _test_():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test_()
    #_test_with_random_values()
    #_test_with_different_parameters()
