from os.path import isfile
import CatClust, distance
from Feature import Feature
import numpy as np
import sys


def get_legal_dist_measures():
    return ['euclidean', 'sqeuclidean', 'correlation', 'cityblock',
            'relevant_distance', 'weighted_threshold', 'forbes-corr',
            'exp_relevant_distance']


def get_legal_feature_types():
    return ['overlap', 'dist_to_closest', 'forbes-corr']


def _files_are_np(fns):
    is_np = True
    for fn in fns:
        if len(fn) < 5 or fn[-4:] != ".npy":
            is_np = False
            break

    return is_np


def stdout_print(*args):
    for arg in args:
        sys.stdout.write("".join(map(str, arg)) if type(arg) == tuple else "%s" % arg)
    sys.stdout.write("\n")


def _check_asserts(donor_fns, reference_fns, reference_types, dist, linkage_method, feature_type, bin_size, k, lock, join):
    for fn in donor_fns:
        assert isfile(fn), "Donor file %s is not found" % fn

    for fn in reference_fns:
        assert isfile(fn), "Reference file %s is not found" % fn

    assert dist in get_legal_dist_measures(), \
        "Distance measure %s is not in allowed. Check get_legal_dist_measures()" % dist

    assert linkage_method in CatClust.linkage_methods.keys(), \
        "Linkage method %s is not allowed. Choose one supported in list %s" % (linkage_method, CatClust.linkage_methods)

    if type(reference_types) == str:
        assert isfile(reference_types), "Reference set types needs to be passed as a file name reference or a list"
    elif type(reference_types) == list:
        assert len(reference_types) == len(reference_fns),\
            "A one to one mapping between reference set and reference set types are needed"
        for t in reference_types:
            assert type(t) == str, "Reference set types needs to be passed as strings"
    else:
        raise TypeError("Reference set types needs to be passed as a file name reference or a list")

    assert feature_type in get_legal_feature_types(),\
        "Unknown feature type \"%s\" choose one from %s" % (feature_type, get_legal_feature_types())

    if dist == "forbes-corr":
        assert 0 < bin_size < Feature.genome_len, "Bin size has to be bigger than 0 and smaller than %s" % Feature.genome_len

        if bin_size < 100000:
            print("WARNING! Bin size of %s results in a very high number of bins!" % bin_size)

        if not (0.0 <= k <= 1.0):
            print("WARNING! With Forbes Correlation, k is highly advised to be in interval [0, 1]")
            print("         k-values smaller than 0 or larger than 1 will result in unintended effects.")

    assert type(lock) == int and lock >= 0, "Locking number has to be an int, larger or equal than 0, not %s" % lock

    assert type(join) == bool, "Join has to be of type bool, not %s" % type(join)


def cluster(donor_fns, reference_fns, reference_types, dist, linkage_method, feature_type, bin_size=1000000, k=0.1, lock=3, join=False, print_progress=True):
    """
    :param donor_fns: list of string with file names to each donor
    :param reference_fns: list of strings with file names to each reference file
    :param reference_types: list of strings of reference types or file name to file with types
    :param dist: string - distance measure to be used
    :param linkage_method: string - linkage method to be used
    :param feature_type: string - how to calculate features
    :param bin_size: optional int - if Forbes Correlation is used, bin size may be changed
    :return: linkage matrix, all categories used to form a cluster in each step
             and the tree representation of all end clusters in linkage (CatClust.Tree object).


    Donor files should be tab separated with one point for each line on format:
    chromosome position

    Reference files should be tab separated with one segment/interval for each line on format:
    chromosome start_pos end_pos

    """
    if print_progress:
        write = stdout_print
    else:
        write = lambda h: None

    _check_asserts(donor_fns, reference_fns, reference_types, dist, linkage_method, feature_type, bin_size, k, lock, join)

    write("CC: Checks passed")
    if _files_are_np(donor_fns):
        donors = np.zeros(len(donor_fns), dtype=object)
        for i, fn in zip(range(donors.shape[0]), donor_fns):
            donors[i] = np.load(fn)
        write("CC: Donors loaded as numpy arrays")
    else:
        donors = Feature.read_donor_tracks(donor_fns)
        write("CC: Donors loaded as text files")

    if _files_are_np(reference_fns):
        ref_set = np.zeros(len(reference_fns), dtype=object)
        for i, fn in zip(range(ref_set.shape[0]), reference_fns):
            ref_set[i] = np.load(fn)
        write("CC: Reference set loaded as numpy arrays")
    else:
        ref_set = Feature.read_feature_tracks_wo_types(reference_fns)
        write("CC: Reference set loaded as text files")

    if type(reference_types) == str:
        with open(reference_types) as f:
            reference_types = [t.rstrip("\n") for t in f.readlines()]
        write("CC: Reference types loaded")

    if feature_type == "overlap":
        X = []
        for donor in donors:
            X.append(np.min(Feature.fast_min_dist(donor, ref_set), axis=0))
        X = np.array(X)
        X = np.invert(X.astype(bool))

        X = Feature.split_features(X, reference_types)
    elif feature_type == "dist_to_closest":
        X = []
        for donor in donors:
            X.append(np.min(Feature.fast_min_dist(donor, ref_set),axis=0))
        X = np.array(X)

        X = Feature.split_features(X, reference_types)
        # Convert to similarity
        X = 1 / (X + 1)
    elif feature_type == "forbes-corr":
        total_bins = np.sum([int(chrom_len / bin_size) + 1 if chrom_len % bin_size != 0 else int(chrom_len / bin_size) for chrom_len in Feature.chrom_len])
        bin_pres = np.zeros((len(donors), len(ref_set), total_bins))

        for donor,i  in zip(donors, range(len(donors))):
            for ref in ref_set:
                bin_pres[i] = Feature.get_bin_feature_vector(donor, ref, bin_size)

        corr = Feature.get_corr_coeff(donors, ref_set)
    else:
        raise Exception("Feature type %s not supported." % feature_type)
    write("CC: Features created with %s" % feature_type)

    if dist == "forbes-corr":
        dM = distance.forbes_corr_pdist(bin_pres, corr, k)
        dM = Feature.group_dm_on_predefined_types(dM, reference_types)
    else:
        dM = []
        for x in X:
            dM.append(distance.pdist(x, metric=dist))
        dM = np.array(dM)

    write("CC: Distance matrix created")

    cat_idx = [np.where(t == reference_types)[0] for t in np.unique(reference_types)]

    write("CC: Starting clustering")
    Z, cc, trees = CatClust.linkage(X=None, cat_idx=cat_idx, method=linkage_method, lock_to=lock, dM=dM, join=join)
    write("CC: Clustering finished")

    return Z, cc, trees
