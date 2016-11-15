__author__ = 'krilange'
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
from collections import defaultdict as dd

from time import time


class Feature:
    """
    A collection of methods to construct arrays of vectors to be used to cluster objects
    """

    all_chroms = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                  '14', '15', '16', '17', '18', '19', '20', '21', '22', 'x', 'y']

    chrom_len = np.array([247199719, 242751149, 199446827, 191263063, 180837866, 170896993,
                          158821424, 146274826, 140442298, 135374737, 134452384, 132289534,
                          114127980, 106360585, 100338915, 88822254, 78654742, 76117153,
                          63806651, 62435965, 46944323, 49528953, 154913754, 57741652])

    genome_len = 3079843747

    @staticmethod
    def make_overlap_array(points, segments):
        """
        >>> points = {0:[0,20,100]}
        >>> segments = [{0:[[0,20],[25,70]]}]
        >>> Feature().make_overlap_array(points,segments)
        array([[ True, False, False]], dtype=bool)

        >>> points = {0:[10]}
        >>> segments = [{0:[[20,40]], 1:[[0,20]]}]
        >>> Feature().make_overlap_array(points,segments)
        array([[False]], dtype=bool)

        >>> points = {0:[10]}
        >>> segments = [{0:[[20,40]]}, {0:[[0,20]]}]
        >>> Feature().make_overlap_array(points,segments)
        array([[False],
               [ True]], dtype=bool)

        Returns an 2D array with booleans, representing where
        the points in the points dict overlaps with any of the segments in the segment dict

        :param points:   dict - keys: region - values: 1D np.array with ints
        :param segments: array of dict - keys: region - values: 2D np.array with ints
        :return: 2D np.array with booleans
        """

        num_objects = sum(len(x) for x in points.values())

        # Make numpy boolean array to represent overlaps
        overlaps = np.zeros((len(segments), num_objects), dtype=bool)

        # Calculate overlaps for all the tracks against ssm-track
        for i in range(len(overlaps)):
            overlaps[i] = Feature.overlap(points, segments[i])

        return overlaps

    @staticmethod
    def overlap(point_dict, segment_dict):
        """
        :param point_dict:   dict - keys: region - values: 1D np.array with ints
        :param segment_dict: dict - keys: region - values: 2D np.array with ints

         returns an 1D boolean array which represent
         if the points in points is within any interval in segments


         NB! np.searchsorted might simplify this algorithm
        """

        total_points = sum(len(x) for x in point_dict.values())
        overlaps = np.zeros(total_points, dtype=bool)

        # Be certain that the keys comes in the same order every time
        keys = list(point_dict.keys())
        keys.sort()

        start = 0
        end = 1

        prev_idx = 0

        for region in keys:
            if region not in segment_dict.keys():
                continue

            points = point_dict[region]
            segments = segment_dict[region]

            if len(segments) > 0 and len(points) > 0:
                i = 0
                for i in range(len(points)):

                    left = 0
                    right = len(segments)
                    mid = right / 2

                    # If point position is less than any interval position or
                    #    point position is greater than any interval position,
                    # skip searching for overlap
                    if not (points[i] <= segments[0][start] or points[i] >= segments[-1][end]):
                        alt_mod = False
                        while True:
                            if segments[mid][start] <= points[i] < segments[mid][end]:
                                # Point is within interval at index mid
                                overlaps[prev_idx + i] = True
                                break
                            elif segments[mid - 1][end] <= points[i] < segments[mid][start]:
                                # Point is between interval at index mid -1 and interval at index mid
                                break
                            else:
                                if segments[mid - 1][end] > points[i]:
                                    right = mid
                                else:
                                    left = mid

                                mid = left + int((right - left) / 2)

                                if alt_mod:
                                    mid += (right - left) % 2
                                    alt_mod = False
                                else:
                                    alt_mod = True

                prev_idx += i

        return overlaps

    @staticmethod
    def get_tp_fp(point_dict, segment_dict):
        tp = 0
        fp = 0

        # Be certain that the keys comes in the same order every time
        keys = point_dict.keys()
        keys.sort()

        for chrom in keys:
            if chrom not in segment_dict.keys():
                fp += len(point_dict[chrom])
            else:
                segment = segment_dict[chrom].T
                segment_start = segment[0]
                segment_end = segment[1]

                for point in point_dict[chrom]:
                    start = segment_start.searchsorted(point, side='right')
                    if segment_end[start - 1] > point:
                        tp += 1
                    else:
                        fp += 1

        return tp, fp


    @staticmethod
    def fast_min_dist(point_dict, segment_dicts):
        total_points = np.sum([len(v) for v in point_dict.values()])
        dist = np.zeros((total_points, len(segment_dicts)), dtype=int)
        dist[:,:] = 2147483647

        # Be certain that the keys comes in the same order every time
        keys = np.sort(list(point_dict.keys()))

        for i in range(segment_dicts.shape[0]):
            chrom_pos = 0
            for chrom in keys:
                if chrom in segment_dicts[i].keys():
                    segment = segment_dicts[i][chrom].T
                    segment_start = segment[0]
                    segment_end = segment[1]

                    for j in range(point_dict[chrom].shape[0]):
                        start = segment_start.searchsorted(point_dict[chrom][j], side='right')

                        # Assertions
                        # point_dict[chrom][j] >= segment_start[start-1]
                        # point_dict[chrom][j] <  segment_start[start]
                        # segment_start[start] <  segment_end[start]

                        if start == segment_start.shape[0]:
                            if segment_start[start-1] <= point_dict[chrom][j] < segment_end[start-1]:
                                dist[j + chrom_pos][i] = 0
                            else:
                                dist[j + chrom_pos][i] = point_dict[chrom][j] - segment_end[start - 1] + 1
                        elif start == 0:
                            dist[j + chrom_pos][i] = segment_start[0] - point_dict[chrom][j]
                        elif segment_end[start - 1] > point_dict[chrom][j]:
                            dist[j + chrom_pos][i] = 0
                        else:
                            if segment_start[start-1] <= point_dict[chrom][j] < segment_end[start-1]:
                                dist[j + chrom_pos][i] = 0
                            else:
                                dist[j + chrom_pos][i] = min(abs(point_dict[chrom][j] - segment_end[start -1] + 1)
                                                     ,abs(segment_start[start] - point_dict[chrom][j]))

                chrom_pos += len(point_dict[chrom])
        return dist

    @staticmethod
    def donors_overlaps(point_dicts, segment_dicts):
        overlaps = [None for _ in range(len(point_dicts))]

        for i in range(len(point_dicts)):
            overlaps[i] = Feature.fast_overlap(point_dicts[i], segment_dicts)

        return overlaps

    @staticmethod
    def fast_overlap(point_dict, segment_dicts):
        total_seg_lens = [np.sum([len(v) for v in segment_dict.values()]) for segment_dict in segment_dicts]
        #overlaps = np.zeros((total_points, len(segment_dicts)), dtype=bool)
        overlaps = [[False for _ in range(seg_len)] for seg_len in total_seg_lens]

        # Be certain that the keys comes in the same order every time
        keys = np.sort(list(point_dict.keys()))

        for i in range(segment_dicts.shape[0]):
            for chrom in keys:
                if chrom in segment_dicts[i].keys():
                    segment = segment_dicts[i][chrom].T
                    segment_start = segment[0]
                    segment_end = segment[1]

                    for j in range(point_dict[chrom].shape[0]):
                        start = segment_start.searchsorted(point_dict[chrom][j], side='right')
                        if segment_end[start - 1] > point_dict[chrom][j]:
                            overlaps[i][start - 1] = True
        return overlaps

    @staticmethod
    def save_tp_fp(donor_tracks, segment_tracks):

        start_time = time()

        for i in range(len(donor_tracks)):
            print("tracks %d/%d" % (i + 1, len(donor_tracks)))
            tp_fp = np.zeros(len(segment_tracks), dtype=np.dtype([('tp', int), ('fp', int)]))

            for j in range(len(segment_tracks)):
                tp_fp[j] = Feature.get_tp_fp(donor_tracks[i], segment_tracks[j])

            np.save("donor_tp_fp/donor_segment_tp_fp_%d" % (i + 1), tp_fp, allow_pickle=False)

            print("Ran for %.2fm. Estemated %.2fm left." % (
            (time() - start_time) / 60, (((time() - start_time) / (i + 1)) * len(donor_tracks)) / 60))

    @staticmethod
    def make_inverse_distance_array(c_track, f_tracks, scale=lambda h: h):

        num_objects = sum(len(x) for x in c_track.values())

        # Make float array for inverse distance
        inv_dist = np.zeros(num_objects * len(f_tracks), dtype=float)
        inv_dist = inv_dist.reshape((len(f_tracks), num_objects))

        # Calculate inverse distance for all the feature tracks against clustering track
        for i in range(len(inv_dist)):
            inv_dist[i] = Feature.inverse_distance(c_track, f_tracks[i], scale)
            print("Made distance array %d of %d" % (i, len(inv_dist)))
        return inv_dist.T

    @staticmethod
    def inverse_distance(track_a, track_b, scale):
        """

        >>> points = {0:[0,20,100]}
        >>> segments = {0:[[0,20],[25,70]]}
        >>> Feature().inverse_distance(points,segments,20,lambda h: h)
        array([ 1.  ,  0.95,  0.  ])

        >>> points = {0:[0,20,100]}
        >>> segments = {0:[[0,20],[25,70]]}
        >>> Feature().inverse_distance(points,segments,20,lambda h: h**3)
        array([ 1.      ,  0.857375,  0.      ])

        :param track_a: dictionary with 1D array like - points
        :param track_b: dictionary with 2D array like - segments where [:,0] = starts and [:,1] = ends
        :param scale: a function to scale the similarity from distance to 0
        :return:
        """
        total_points = sum(len(x) for x in track_a.values())
        distance = np.zeros(total_points, dtype=float) + 1

        # Be certain that the keys comes in the same order every time
        keys = np.sort(list(track_a.keys()))

        start = 0
        end = 1

        prev_idx = 0

        for region in keys:
            if region not in track_b.keys():
                continue

            points = track_a[region]
            segments = track_b[region]

            # print "Points(%d)[%s]: %s" % (len(points), region, points)
            # print "Segments(%d)[%s]: %s" % (len(segments), region, segments)

            if len(segments) > 0 and len(points) > 0:
                i = 0
                for i in range(len(points)):

                    left = 0
                    right = len(segments)
                    mid = right / 2

                    if points[i] <= segments[0][start]:
                        # If point position is less than any interval position
                        distance[prev_idx + i] = segments[0][start] - points[i]
                    elif points[i] >= segments[-1][end]:
                        # If point position is greater than any interval position
                        distance[prev_idx + i] = points[i] - segments[-1][end] + 1
                    else:
                        alt_mod = False
                        while True:
                            if segments[mid][start] <= points[i] < segments[mid][end]:
                                # Point is within interval at index mid
                                distance[prev_idx + i] = 0
                                break
                            elif segments[mid - 1][end] <= points[i] < segments[mid][start]:
                                # Point is between interval at index mid -1 and interval at index mid
                                distance[prev_idx + i] = min((points[i] - segments[mid - 1][end] + 1),
                                                             (segments[mid][start] - points[i]))
                                break
                            else:
                                if segments[mid - 1][end] > points[i]:
                                    right = mid
                                else:
                                    left = mid

                                mid = left + (right - left) / 2

                                if alt_mod:
                                    mid += (right - left) % 2
                                    alt_mod = False
                                else:
                                    alt_mod = True

                prev_idx += i

        return np.asarray([scale(Feature.dist_to_sim(dist)) for dist in distance])

    @staticmethod
    def make_inverse_distance_array_with_points(c_track, f_tracks, limit, scale=lambda h: h):

        num_objects = sum(len(x) for x in c_track.values())

        # Make float array for inverse distance
        inv_dist = np.zeros(num_objects * len(f_tracks), dtype=float)
        inv_dist = inv_dist.reshape((len(f_tracks), num_objects))

        # Calculate inverse distance for all the feature tracks against clustering track
        for i in range(len(inv_dist)):
            for region in f_tracks[i].keys():
                if len(f_tracks[i][region]) > 0:
                    f_tracks[i][region] = np.average(f_tracks[i][region], axis=1)
            inv_dist[i] = Feature.inverse_distance_with_points(c_track, f_tracks[i], limit, scale)

        return inv_dist.transpose()

    @staticmethod
    def inverse_distance_with_points(track_a, track_b, limit, scale):
        """
        >>> points = {0:[]}
        >>> segments = {0:[]}
        >>> Feature().inverse_distance_with_points(points,segments,20,lambda h: h)
        array([], dtype=float64)

        >>> points = {0:[1,39,70]}
        >>> segments = {0:[20]}
        >>> Feature().inverse_distance_with_points(points,segments,20,lambda h: h)
        array([ 0.05,  0.05,  0.  ])

        >>> points = {0:[0,30,40]}
        >>> segments = {0:[0,20,70]}
        >>> Feature().inverse_distance_with_points(points,segments,20,lambda h: h)
        array([ 1. ,  0.5,  0. ])

        >>> points = {0:[0,20,80]}
        >>> segments = {0:[10,20,75]}
        >>> Feature().inverse_distance_with_points(points,segments,20,lambda h: h)
        array([ 0.5 ,  1.  ,  0.75])

        >>> points = {0:[0,20,80]}
        >>> segments = {0:[10,20,75]}
        >>> Feature().inverse_distance_with_points(points,segments,20,lambda h: h**3)
        array([ 0.125   ,  1.      ,  0.421875])

        #>>> points = {0:[0,20,100]}
        #>>> segments = {0:[[0,20],[25,70]]}
        #>>> Feature().inverse_distance_with_points(points,segments,20,lambda h: h**3)
        sdfsdf

        :param track_a: dictionary with 1D array like - points
        :param track_b: dictionary with 2D array like - segments where [:,0] = starts and [:,1] = ends
        :param limit: the distance where the similarity should be 0
        :param scale: a function to scale the similarity from distance to 0
        :return:
        """
        total_points = sum(len(x) for x in track_a.values())
        distance = np.zeros(total_points, dtype=float) + limit + 1

        # Be certain that the keys comes in the same order every time
        keys = track_a.keys()
        keys.sort()

        prev_idx = 0

        for region in keys:
            points_a = track_a[region]
            points_b = track_b[region]

            if len(points_b) > 0 and len(points_a) > 0:
                i = 0
                for i in range(len(points_a)):

                    left = 0
                    right = len(points_b)
                    mid = right / 2

                    if points_a[i] <= points_b[0]:
                        # If point_a is less than any point_b
                        distance[prev_idx + i] = points_b[0] - points_a[i]
                    elif points_a[i] >= points_b[-1]:
                        # If point_a i grater than the last point in points_b
                        distance[prev_idx + i] = points_a[i] - points_b[-1]
                    else:
                        alt_mod = False

                        while True:
                            # found shortest distance
                            if points_b[mid] >= points_a[i] >= points_b[mid - 1]:
                                distance[prev_idx + i] = min(points_b[mid] - points_a[i],
                                                             points_a[i] - points_b[mid - 1])
                                break

                            if points_a[i] < points_b[mid]:
                                right = mid
                            else:
                                left = mid

                            mid = left + (right - left) / 2

                            if alt_mod:
                                mid += (right - left) % 2

                            alt_mod = not alt_mod

                prev_idx += i

        return np.asarray([scale(Feature.dist_to_sim(dist)) for dist in distance])

    @staticmethod
    def dist_to_sim(dist):
        return 1 / (1 + abs(float(dist)))

    @staticmethod
    def get_sim_matrix(choices, c_track_dict, f_track_dict, concat_fun=np.max):
        sim = None
        import time

        t = time.time()

        if choices.input['Cluster multiple tracks']:
            sim = np.zeros((len(c_track_dict), len(f_track_dict)), dtype=float)
            if choices.f_type == 'inverse distance':
                if choices.extra['Convert to points']:
                    for i in range(len(c_track_dict)):
                        print("inverse distance array %d of %d" % (i + 1, len(c_track_dict)))
                        sim[i] = concat_fun(
                            Feature.make_inverse_distance_array_with_points(c_track_dict[i], f_track_dict,
                                                                            int(choices.limit)), axis=0)
                else:
                    for i in range(len(c_track_dict)):
                        print("inverse distance array %d of %d" % (i + 1, len(c_track_dict)))
                        sim[i] = concat_fun(
                            Feature.make_inverse_distance_array(c_track_dict[i], f_track_dict, int(choices.limit)),
                            axis=0)
            elif choices.f_type == 'overlap':
                for i in range(len(c_track_dict)):
                    print("overlap array %d of %d" % (i + 1, len(c_track_dict)))
                    sim[i] = concat_fun(Feature.make_overlap_array(c_track_dict[i], f_track_dict), axis=1)
        else:
            if choices.f_type == 'inverse distance':
                if choices.extra['Convert to points']:
                    sim = Feature.make_inverse_distance_array_with_points(c_track_dict, f_track_dict,
                                                                          int(choices.limit))
                else:
                    sim = Feature.make_inverse_distance_array(c_track_dict, f_track_dict, int(choices.limit))

            elif choices.f_type == 'overlap':
                sim = Feature.make_overlap_array(c_track_dict, f_track_dict)

        print("sim matrix creation time %s\n" % (time.time() - t))

        return sim

    @staticmethod
    def _get_overlap_chr(points, segments):
        overlaps = np.zeros(len(points), dtype=bool)

        # Convenient variables
        start = 0
        end = 1

        prev_idx = 0

        if len(segments) > 0 and len(points) > 0:
            # Make interpreter/compiler happy
            i = 0
            for i in range(len(points)):
                # Initial search space
                left = 0
                right = len(segments)
                mid = int(right / 2)

                # If point position is less than any interval position or
                # point position is greater than any interval position,
                # skip searching for overlap. (result = false, and arr init = false)
                if not (points[i] <= segments[0][start] or points[i] >= segments[-1][end]):
                    alt_mod = False
                    while True:
                        if segments[mid][start] <= points[i] < segments[mid][end]:
                            # Point is within interval at index mid
                            overlaps[prev_idx + i] = True
                            break
                        elif segments[mid - 1][end] <= points[i] < segments[mid][start]:
                            # Point is between interval at index mid -1 and interval at index mid
                            # overlaps[prev_idx + 1] = False
                            break
                        else:
                            # Adjust search space
                            if segments[mid - 1][end] > points[i]:
                                right = mid
                            else:
                                left = mid

                            mid = left + int((right - left) / 2)

                            if alt_mod:
                                mid += (right - left) % 2
                                alt_mod = False
                            else:
                                alt_mod = True

            prev_idx += i

        return overlaps

    @staticmethod
    def _get_binned_overlap_fraction(overlaps, pos, bin_size, chrom_len):
        bins = np.zeros(int(chrom_len / bin_size) + 1 if chrom_len % bin_size != 0 else int(chrom_len / bin_size))
        pos = np.array(pos)

        start = 0
        for i in range(0, bins.shape[0]):
            idx = np.where(pos[start:] < bin_size * (i + 1))[0]
            num_overlaps = overlaps[start:][idx].sum()

            if idx.shape[0] == 0:
                # No points in bin
                bins[i] = np.nan
            elif num_overlaps == 0:
                # Points, but no overlaps in bin
                bins[i] = 0
            else:
                # Some overlaps in bin
                bins[i] = num_overlaps / idx.shape[0]

            if idx.shape[0] > 0:
                start += idx[-1] + 1

            if start == pos.shape[0]:
                # No more position | end of computations
                if i + 1 < bins.shape[0]:
                    bins[i + 1:] = np.nan
                break

        return bins

    @staticmethod
    def get_bin_feature_vector(points_dict, segments_dict, bin_size):
        overlaps = {}

        # Calculate overlaps per chromosome
        for chrom in points_dict.keys():
            if chrom in segments_dict.keys():
                overlaps[chrom] = Feature._get_overlap_chr(points_dict[chrom], segments_dict[chrom])
            else:
                overlaps[chrom] = np.zeros(len(points_dict[chrom]), dtype=bool)

        bins = np.zeros(len(Feature.all_chroms), dtype=object)
        for i in range(len(Feature.all_chroms)):
            chrom = Feature.all_chroms[i]
            chrom_len = Feature.chrom_len[i]

            if chrom in points_dict.keys():
                bins[i] = Feature._get_binned_overlap_fraction(overlaps[chrom], points_dict[chrom], bin_size, chrom_len)
            else:
                bins[i] = np.zeros(
                    int(chrom_len / bin_size) + 1 if chrom_len % bin_size != 0 else int(chrom_len / bin_size))
                bins[i][:] = np.nan

        return np.array([item for sublist in bins for item in sublist])

    @staticmethod
    def read_donor_tracks(fns):
        """
        Takes fns of donors and put them into a list of dicts,
        where keys in each dict is chromosome
        """
        donors = []

        for i, fn in zip(range(len(fns)), fns):
            donors.append(dd(list))
            with open(fn) as f:
                for line in f:
                    chrom, pos = line.split("\t")
                    donors[-1][chrom].append(int(pos))
            donors[-1] = dict(donors[-1])

        for i in range(len(donors)):
            for key in donors[i].keys():
                donors[i][key] = np.array(donors[i][key])

        return np.array(donors)

    @staticmethod
    def load_donor_tracks(fns):
        donors = []

        for i, fn in zip(range(len(fns)), fns):
            print("...loading track %d/%d: %s" % (i + 1, len(fns), fn))
            x = np.load(fn)

            dd = {}
            for chrom in np.unique(x['chr']):
                dd[chrom] = x[x['chr'] == chrom]['pos']

            donors.append(dd)

        return donors

    @staticmethod
    def read_feature_tracks(fns, type_map):
        features = []
        feature_types = []

        for i, fn in zip(range(len(fns)), fns):
            features.append(dd(list))
            with open(fn) as f:
                lower_fn = fn.lower()
                for e in type_map:
                    if e[0] in lower_fn:
                        feature_types.append(e[1])
                        break

                for line in f:
                    chrom, start, end = line.split("\t")
                    features[-1][chrom].append([int(start), int(end)])

        return features, feature_types

    @staticmethod
    def read_feature_tracks_wo_types(fns):
        features = []

        for i, fn in zip(range(len(fns)), fns):
            features.append(dd(list))
            with open(fn) as f:
                for line in f:
                    chrom, start, end = line.split("\t")
                    features[-1][chrom].append([int(start), int(end)])
            features[-1] = dict(features[-1])

        for i in range(len(features)):
            for key in features[i].keys():
                features[i][key] = np.array(features[i][key])

        return np.array(features)

    @staticmethod
    def strip_feature_files(fns, chrom=0, start=1, end=2, skiplines=1):
        from os import rename, remove

        for fn in fns:
            with open(fn) as f:
                with open(fn + ".tmp", 'w') as tmp:
                    for i in range(skiplines):
                        f.readline()

                    for line in f:
                        line = line.split("\t")
                        if len(line[chrom]) > 3:
                            line[chrom] = line[chrom][3:]
                        tmp.write("%s\t%d\t%d\n" % (line[chrom], int(line[start]), int(line[end])))

            remove(fn)
            rename(fn + ".tmp", fn)

    @staticmethod
    def read_map_file(fn):
        maps = []
        with open(fn) as f:
            for line in f:
                line = line.split("\t")
                maps.append([line[0], line[1][:-1]])

        return maps

    @staticmethod
    def create_extended_feature_vectors(feature_list, donor_list, bin_size=1000000):
        donor_matrixs = []

        for i, donor in zip(range(len(donor_list)), donor_list):
            donor_matrix = []
            print("...creating features for donor %d/%d" % (i + 1, len(donor_list)))
            for feature in feature_list:
                donor_matrix.append(Feature.get_bin_feature_vector(donor, feature, bin_size))
            donor_matrixs.append(donor_matrix)

        return donor_matrixs

    @staticmethod
    def merge_features(donor_feature_matrixs, types):
        donor_feature_matrixs = np.asarray(donor_feature_matrixs)
        unique_types = np.unique(types)
        merged_donor_matrixs = []

        for donor_matrix in donor_feature_matrixs:
            merged_donor_matrix = []
            for t in unique_types:
                idxs = np.where(t == types)[0]
                merged_donor_matrix.append(donor_matrix[idxs].sum(0) / len(idxs))
            merged_donor_matrixs.append(merged_donor_matrix)

        return np.array(merged_donor_matrixs), unique_types

    @staticmethod
    def split_features(donor_features, types):
        donor_features = np.asarray(donor_features)
        types = np.asarray(types)
        unique_types = np.unique(types)
        split_features = np.zeros(unique_types.shape[0], dtype=object)

        for i, t in zip(range(unique_types.shape[0]), unique_types):
            idxs = np.where(t == types)[0]
            split_features[i] = donor_features[:,idxs]

        return split_features

    @staticmethod
    def get_total_overlap(fts):
        overlap_lens = []
        for ft in fts:
            ft_overlap = 0
            for key in ft.keys():
                if len(ft[key]) >= 0:
                    chr_f = np.array(ft[key])
                    ft_overlap += (chr_f[::, 1] - chr_f[::, 0]).sum()
            overlap_lens.append(float(ft_overlap) / Feature.genome_len)

        return overlap_lens

    @staticmethod
    def total_overlap_to_weights(ol):
        ol = np.array(ol)
        high = np.max(ol)
        base = np.average(ol) / high
        top = np.average(ol) / np.min(ol)

        # Weight from base to high
        weights = base + ((high - ol) / np.max(high - ol)) * (top - base)
        # Scale weights from min(weights)/max(weights) to 1
        return weights / np.max(weights)

    @staticmethod
    def read_donor_features(base_dir, limit_donors=0, limit_features=0):
        if base_dir[-1] == "/":
            base_dir = base_dir[:-1]

        donor_ids = [d for d in listdir(base_dir) if isdir("%s/%s" % (base_dir, d))]
        if limit_donors > 0:
            donor_ids = donor_ids[0:min(limit_donors, len(donor_ids))]
        feature_fns = [f for f in listdir("%s/%s" % (base_dir, donor_ids[0])) if
                       isfile(join("%s/%s/" % (base_dir, donor_ids[0]), f))]
        if limit_features > 0:
            feature_fns = feature_fns[0:min(limit_features, len(feature_fns))]

        with open("%s/%s/%s" % (base_dir, donor_ids[0], feature_fns[0])) as f:
            bin_len = sum([1 for _ in f])

        donor_features = np.zeros((len(donor_ids), len(feature_fns), bin_len))

        for i, donor_id in zip(range(len(donor_ids)), donor_ids):
            print("Reading feature data for donor %d/%d: %s" % (i + 1, len(donor_ids), donor_id))
            for j, feature_fn in zip(range(len(feature_fns)), feature_fns):
                with open("%s/%s/%s" % (base_dir, donor_ids[i], feature_fns[j])) as f:
                    donor_features[i, j] = np.array(f.readlines())

        return donor_features

    @staticmethod
    def read_total_overlaps(fn):
        ol = []
        fns = []

        with open(fn) as f:
            for l in f:
                l = l.split("\t")
                ol.append(float(l[1]))
                fns.append(l[0])

        return np.array(ol), fns

    @staticmethod
    def preprocessed_feature_data_to_dM(limit_donors=0, limit_features=0):
        """
        Reading from cached system memory (RAM - DDR3 1600MHz)
        ===============================
        Data set size: (25, 1385, 3091)
        Reading and manipulating preprocessed time: 390.707517
        dM creation time: 1048.870779
        ~= 24 min.

        """
        from time import time

        df = Feature.read_donor_features("donor_features", limit_donors, limit_features)

        ol, ffns = Feature.read_total_overlaps("donor_features/total_overlaps")

        if limit_features == 0:
            limit_features = len(ol)

        ol = ol[:limit_features]
        weights = Feature.total_overlap_to_weights(ol)

        ffns = ffns[:limit_features]

        with open("donor_features/types") as f:
            types = [l.strip("\n") for l in f][:limit_features]

        df, types = Feature.merge_features(df, types)

        # Swap axis to have types at first index
        df = np.swapaxes(df, 0, 1)

        dM = Feature.make_extended_dm(df, weights)

        return dM, types

    @staticmethod
    def _get_bin_presence(pos, bin_size, chrom_len):
        bins = np.zeros(int(chrom_len / bin_size) + 1 if chrom_len % bin_size != 0 else int(chrom_len / bin_size),
                        dtype=bool)
        pos = np.array(pos)

        start = 0
        for i in range(0, bins.shape[0]):

            idx = np.where(pos[start:] < bin_size * (i + 1))[0]

            if idx.shape[0] > 0:
                bins[i] = True
                start += idx[-1] + 1

            if start == pos.shape[0]:
                break

        return bins

    @staticmethod
    def get_bin_presences(donor_tracks, bin_size):

        bin_sizes = [int(chrom_len / bin_size) + 1 if chrom_len % bin_size != 0 else int(chrom_len / bin_size) for
                     chrom_len in Feature.chrom_len]
        total_bins = sum(bin_sizes)

        chrom_hits = dd(int)

        bins = np.zeros((len(donor_tracks), total_bins), dtype=bool)

        for i, donor in zip(range(len(donor_tracks)), donor_tracks):
            for j in range(len(Feature.all_chroms)):
                chrom = Feature.all_chroms[j]
                chrom_len = Feature.chrom_len[j]

                if chrom in donor.keys():
                    chrom_hits[chrom] += 1
                    chr_bin = Feature._get_bin_presence(donor[chrom], bin_size, chrom_len)
                    bins[i][sum(bin_sizes[:j]):sum(bin_sizes[:j + 1])] = chr_bin

        return bins, chrom_hits

    @staticmethod
    def _findAllStartAndEndEvents(t1s, t1e, t2s, t2e):
        # assert no overlaps..
        # create arrays multiplied by 8 to use last three bits to code event type,
        # Last three bits: relative to 4 (100): +/- 1 for start/end of track1, +/- 2 for track2..
        t1CodedStarts = t1s * 8 + 5
        t1CodedEnds = t1e * 8 + 3
        t2CodedStarts = t2s * 8 + 6
        t2CodedEnds = t2e * 8 + 2

        allSortedCodedEvents = np.concatenate((t1CodedStarts, t1CodedEnds, t2CodedStarts, t2CodedEnds))
        allSortedCodedEvents.sort()

        allEventCodes = (allSortedCodedEvents % 8) - 4

        # "//" Fix for Python 3.5 where int division gives float and "//" is used for floored division
        allSortedDecodedEvents = allSortedCodedEvents // 8

        allEventLengths = allSortedDecodedEvents[1:] - allSortedDecodedEvents[:-1]

        # due to the coding, the last bit now has status of track1, and the second last bit status of track2
        # thus, 3 is cover by both, 2 is cover by only track2, 1 is cover by only track1, 0 is no cover
        # this works as there are no overlaps, and bits will thus not "spill over"..
        cumulativeCoverStatus = np.add.accumulate(allEventCodes)

        return allSortedDecodedEvents, allEventLengths, cumulativeCoverStatus

    @staticmethod
    def _computeRawOverlap(t1s, t1e, t2s, t2e, binSize):
        allSortedDecodedEvents, allEventLengths, cumulativeCoverStatus = Feature._findAllStartAndEndEvents(t1s, t1e,
                                                                                                           t2s, t2e)

        tn, fp, fn, tp = [(allEventLengths[cumulativeCoverStatus[:-1] == status]).sum() for status in range(4)]

        if len(allSortedDecodedEvents) > 0:
            tn += allSortedDecodedEvents[0] + (binSize - allSortedDecodedEvents[-1])
        else:
            tn += binSize

        return tn, fp, fn, tp

    @staticmethod
    def computeRawOverlap(t1s, t1e, p, binSize):
        """
        >>> binSize = 100
        >>> t1s = np.array([13, 34, 55, 84, 91])
        >>> t1e = np.array([30, 51, 81, 90, 92])
        >>> p = np.array([5,34, 81, 87, 95])
        >>> tp = 2
        >>> fp = 3
        >>> tn = binSize - (t1e - t1s).sum() - fp
        >>> fn = (t1e - t1s).sum() - tp
        >>> Feature.computeRawOverlap(p, p+1, t1s, t1e, binSize)
        (30, 3, 65, 2)
        """
        return Feature._computeRawOverlap(p, p + 1, t1s, t1e, binSize)

    @staticmethod
    def get_corr_coeff(point_dicts, segment_dicts):
        c = []
        tn, fp, fn, tp = (0, 1, 2, 3)
        for i in range(len(point_dicts)):
            coeffs = np.zeros(len(segment_dicts), dtype=float)
            for j in range(len(segment_dicts)):
                res = np.zeros(4, dtype=np.int64)
                for chrom, chrom_len in zip(Feature.all_chroms, Feature.chrom_len):
                    if chrom in point_dicts[i].keys() and chrom in segment_dicts[j].keys():
                        p = point_dicts[i][chrom]
                        t1s, t1e = segment_dicts[j][chrom].T
                        chrom_res = Feature.computeRawOverlap(t1s, t1e, p, chrom_len)
                        res += chrom_res
                    elif chrom in point_dicts[i].keys():
                        res[fp] += len(point_dicts[i][chrom])
                        res[tn] += chrom_len - len(point_dicts[i][chrom])
                    elif chrom in segment_dicts[j].keys():
                        res[fn] += len(segment_dicts[j][chrom])
                        res[tn] += chrom_len - len(segment_dicts[j][chrom])
                    else:
                        res[tn] += chrom_len
                coeffs[j] = (res[tp] * res[tn] - res[fn] * res[fp]) / (
                np.sqrt(res[tp] + res[fn]) * np.sqrt(res[tn] + res[fp]) * np.sqrt(res[tp] + res[fp]) * np.sqrt(
                    res[tn] + res[fn]))

            c.append(coeffs)

        return np.array(c)

    @staticmethod
    def fns_to_types(fns, type_map):
        types = []

        for fn in fns:
            fn = fn.lower()
            for t in type_map:
                if t[0] in fn:
                    types.append(t[1])
                    break
        return np.array(types)

    @staticmethod
    def cluster_all():
        import CatClust

        dm_fns = ["dms/%s" % f for f in listdir("dms/") if isfile(join("dms/", f))]

        #dm_fns = [fn for fn in dm_fns if "1M" in fn]
        dm_fns = [fn for fn in dm_fns if "10M" in fn]

        #dm_fns = [fn for fn in dm_fns if "0.0." in fn]
        #dm_fns = [fn for fn in dm_fns if "0.05." in fn]
        #dm_fns = [fn for fn in dm_fns if "0.1." in fn]
        #dm_fns = [fn for fn in dm_fns if "0.2." in fn]
        #dm_fns = [fn for fn in dm_fns if "0.9." in fn]
        #dm_fns = [fn for fn in dm_fns if "1.0." in fn]

        methods = ['single', 'complete']
        lock_to = [0,1,3]

        for dm_fn in dm_fns:
            bin_size = dm_fn.split("_")[1]
            k = dm_fn.split("_")[2][:-4]
            for method in methods:
                for lock in lock_to:
                    t = time()
                    print("Clustering", method, lock, bin_size, k)
                    #dm = Feature.group_dm_on_types(np.load(dm_fn))
                    dm = np.load(dm_fn)
                    Z, cc, trees = CatClust.linkage(None, range(dm.shape[0]), lock_to=lock, method=method, dM=dm)
                    np.save("Z_%s_l%d%s%s.npy" % (method, lock, bin_size, k), np.array(Z))
                    np.save("cc_%s_l%d%s%s.npy" % (method, lock, bin_size, k), np.array(np.array(cc)))
                    print("Clustering took: ", (time() - t))

    @staticmethod
    def rgba_to_rgb_str(rgba):
        r = hex(int(rgba[0] * 255)).split("x")[-1]
        g = hex(int(rgba[1] * 255)).split("x")[-1]
        b = hex(int(rgba[2] * 255)).split("x")[-1]
        a = hex(int(rgba[3] * 255)).split("x")[-1]
        return ("#%2s%2s%2s" % (r, g, b)).replace(" ", "0")

    @staticmethod
    def group_on_types(arr, type_ids, group_fun=np.sum, dtype=float):
        arr = np.asarray(arr, dtype)
        unique = np.unique(type_ids)
        grouped_arr = np.zeros((arr.shape[0],unique.shape[0]),dtype=dtype)


        for i in range(arr.shape[0]):
            for j in range(unique.shape[0]):
                grouped_arr[i][j] = group_fun(arr[i][unique[j] == type_ids])

        return grouped_arr

    @staticmethod
    def save_all_plots():
        import sys
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from scipy.cluster.hierarchy import dendrogram
        import CatClust

        sys.setrecursionlimit(2000)
        Z_fns = ["res/%s" % f for f in listdir("res/") if isfile(join("res/", f)) and "Z_" in f]

        cmap = plt.get_cmap('gray')

        feature_fns = ["dset/feature/dnasehs/%s" % f for f in listdir("dset/feature/dnasehs/") if
                       isfile(join("dset/feature/dnasehs/", f))]
        type_map = Feature.read_map_file("dset/cell_tissue.txt")
        types = Feature.fns_to_types(feature_fns, type_map)

        unique_types = np.unique(types)
        type_ids = np.zeros(len(types), int)
        for i, t in zip(range(len(unique_types)), unique_types):
            type_ids[np.where(types == t)[0]] = i

        m = cm.ScalarMappable()

        for Z_fn in Z_fns:
            Z = np.load(Z_fn)
            cc = np.load("%s/cc%s" % (Z_fn.split("/")[0], Z_fn.split("/")[1][1:]))

            bot = Z[Z.shape[0]//4,2]
            not_nans = Z[:,2][np.invert(np.isnan(Z[:,2]))]
            if len(not_nans) == 0:
                print(Z)
                print("ALL NANS")
                print(Z_fn)
                continue
            top = np.max(not_nans) + np.sort(not_nans)[-1] - np.sort(not_nans)[-2]
            yax = np.linspace(bot,top,5)

            if len(cc.shape) > 1 or cc.dtype == object:
                primary_cat = np.array([c[0] for c in cc])
            else:
                primary_cat = cc

            colors = [Feature.rgba_to_rgb_str(cmap(i)) for i in np.linspace(0.1, 0.9, np.unique(type_ids).shape[0])]

            d = dendrogram(Z,no_plot=True)

            tree_dict = CatClust.Tree.Z_to_tree_dict(Z, primary_cat)

            cat_counts = np.zeros((len(tree_dict.keys()), 163))
            clust_ids = np.array(list(tree_dict.keys()))

            for i in range(cat_counts.shape[0]):
                cat_counts[i] = tree_dict[i].get_cat_counts_as_arr(163)
                if cat_counts[i].sum() > 0:
                    cat_counts[i] = cat_counts[i] / cat_counts[i].sum()

            cat_counts = Feature.group_on_types(cat_counts, type_ids)

            color_map = np.zeros(Z.shape[0]*2+1,dtype=object)

            gray = "#333333"

            for i in range(cat_counts.shape[0]):
                cat = np.argmax(cat_counts[i])
                if cat_counts[i,cat] == 0:
                    color_map[clust_ids[i]] = gray
                else:
                    color_map[clust_ids[i]] = colors[cat]

            f = plt.figure()
            d = dendrogram(Z, no_labels=True, show_leaf_counts=True,
                           link_color_func=lambda k: "#333333")
            plt.ylim((Z[0,2],top))

            plt.savefig(filename="%s/plot/%s_%s_whole.png" % (Z_fn.split("/")[0], Z_fn.split("/")[1], "plot"),
                        format='png')

            f = plt.figure()
            d = dendrogram(Z, no_labels=True, show_leaf_counts=True,
                           link_color_func=lambda k: "#333333")
            plt.yticks(yax)
            plt.ylim((bot, top))
            print((bot,top))
            plt.savefig(filename="%s/plot/%s_%s_upper.png" % (Z_fn.split("/")[0], Z_fn.split("/")[1], "plot"),
                        format='png')

            plt.close()

    @staticmethod
    def group_dm_on_types(dm, group_fun=np.min):
        feature_fns = ["dset/feature/dnasehs/%s" % f for f in listdir("dset/feature/dnasehs/") if
                       isfile(join("dset/feature/dnasehs/", f))]
        type_map = Feature.read_map_file("dset/cell_tissue.txt")
        types = Feature.fns_to_types(feature_fns, type_map)

        unique_types = np.unique(types)
        type_ids = np.zeros(len(types), int)
        for i, t in zip(range(len(unique_types)), unique_types):
            type_ids[np.where(types == t)[0]] = i

        new_dm = np.zeros((unique_types.shape[0], dm.shape[1]))

        for i in range(unique_types.shape[0]):
            new_dm[i, :] = group_fun(dm[type_ids == i, :], axis=0)

        return new_dm

    @staticmethod
    def group_dm_on_predefined_types(dm, types, group_fun=np.min):
        types = np.asarray(types)
        u_types = np.unique(types)

        type_ids = np.zeros(len(types), int)
        for i, t in zip(range(u_types.shape[0]), u_types):
            type_ids[np.where(types == t)[0]] = i

        new_dm = np.zeros((u_types.shape[0], dm.shape[1]))

        for i in range(u_types.shape[0]):
            new_dm[i, :] = group_fun(dm[type_ids == i, :], axis=0)

        return new_dm

    @staticmethod
    def get_end_cluster_ids(tree):
        uids = []

        for t in list(tree.values()):
            if np.isnan(t.dist):
                if not np.isnan(t.left.dist):
                    uids.append(t.left.id)
                if not np.isnan(t.right.dist):
                    uids.append(t.right.id)
        return np.unique(uids)

    @staticmethod
    def get_matches_by_cats(leaves, cats):
        cats = np.array(cats)

        if type(leaves[0][0]) != np.str_:
            with open("true_prim_type") as f:
                true_labs = [l.rstrip("\n").lower() for l in f]
                true_labs = np.array(true_labs)

                leaf_labels = np.zeros(len(leaves), dtype=object)
                for i in range(leaf_labels.shape[0]):
                    leaf_labels[i] = np.zeros(len(leaves[i]), dtype=np.dtype('<U20'))

                for i in range(len(leaves)):
                    for j in range(len(leaves[i])):
                        leaf_labels[i][j] = true_labs[leaves[i][j]]
        else:
            leaf_labels = leaves

        total_leaves = np.sum([len(l) for l in leaf_labels])
        matches = np.zeros(len(leaves))

        for i in range(len(leaves)):
            matches[i] = np.where(cats[i] == np.array(leaf_labels[i]))[0].shape[0]

        return matches, total_leaves

    @staticmethod
    def get_occ_matrix(leaves):
        with open("true_prim_type") as f:
            true_labs = [l.rstrip("\n").lower() for l in f]
        true_labs = np.array(true_labs)

        leaf_labels = np.zeros(len(leaves), dtype=object)
        for i in range(leaf_labels.shape[0]):
            leaf_labels[i] = np.zeros(len(leaves[i]), dtype=np.dtype('<U20'))

        for i in range(len(leaves)):
            for j in range(len(leaves[i])):
                leaf_labels[i][j] = true_labs[leaves[i][j]]

        unique_leaf_labels = np.unique([k for l in leaf_labels for k in np.unique(l)])

        twcp = np.zeros((leaf_labels.shape[0], unique_leaf_labels.shape[0]))

        for i in range(len(leaves)):
            for j in range(unique_leaf_labels.shape[0]):
                occs = np.where(unique_leaf_labels[j] == np.array(leaf_labels[i]))[0].shape[0]
                twcp[i][j] += occs

        return twcp

    @staticmethod
    def evaluate_clustering_by_cats(z, cc, iterations=1000):
        import CatClust
        tree = CatClust.Tree.Z_to_tree_dict(z, cc)
        end_clusters = Feature.get_end_cluster_ids(tree)

        if len(end_clusters) == 0:
            end_clusters = Feature.cut_tree(tree, 20)

        types = np.load("types.npy")
        cats = []
        for i in range(end_clusters.shape[0]):
            cat_counts = tree[end_clusters[i]].get_cat_counts()
            if len(cat_counts.values()) == 0:
                cats.append("no_cat")
            else:
                cats.append(types[list(cat_counts.keys())[np.argmax(list(cat_counts.values()))]])

        with open("true_prim_type") as f:
            true_labs = [l.rstrip("\n").lower() for l in f]
        true_labs = np.array(true_labs)

        if len(end_clusters) == 0:
            end_clusters = Feature.cut_tree(tree, 20)

        leaves = []
        for i in range(end_clusters.shape[0]):
            leaves.append(tree[end_clusters[i]].get_leaf_ids())

        matches, total_leaves = Feature.get_matches_by_cats(leaves, cats)

        match_per = np.sum(matches)/total_leaves

        flat_leaves = true_labs

        leaf_idxs = [(0, len(leaves[0]))]
        for i in range(1, len(leaves)):
            leaf_idxs.append((leaf_idxs[-1][1], leaf_idxs[-1][1] + len(leaves[i])))

        rand_match_pers = []
        for i in range(iterations):
            np.random.shuffle(flat_leaves)
            leaves = [flat_leaves[leaf_idxs[j][0]:leaf_idxs[j][1]] for j in range(len(leaves))]
            matches, total_leaves = Feature.get_matches_by_cats(leaves, cats)
            rand_match_pers.append(np.sum(matches) / np.sum(total_leaves))

        return ("%.4f,%.4f,%.4f,%.4f" % (match_per, np.average(rand_match_pers),match_per / np.average(rand_match_pers) - 1, np.sum(np.array(rand_match_pers) > match_per) / iterations))

    @staticmethod
    def cut_tree(tree,min_num_end_clusters):
        root = list(tree.values())[-1]
        nodes = [root]
        new_nodes = []
        leaf_nodes = []
        while (np.unique([node.id for node in new_nodes]).shape[0] +  np.unique([node.id for node in leaf_nodes]).shape[0]) < min_num_end_clusters:
            new_nodes = []
            for node in nodes:
                if node.left != None:
                    if node.left.left == None and node.left.right == None:
                        leaf_nodes.append(node.left)
                    else:
                        new_nodes.append(node.left)
                if node.right != None:
                    if node.right.left == None and node.right.right == None:
                        leaf_nodes.append(node.right)
                    else:
                        new_nodes.append(node.right)
            nodes = new_nodes

        end_clusters = [node.id for node in new_nodes] + [node.id for node in leaf_nodes]
        end_clusters = np.unique(end_clusters)

        return end_clusters

    @staticmethod
    def label_and_evaluate_clustering(z, cc,iterations=10000,scaled=True):
        import CatClust
        tree = CatClust.Tree.Z_to_tree_dict(z, cc)
        end_clusters = Feature.get_end_cluster_ids(tree)

        if len(end_clusters) == 0:
            end_clusters = Feature.cut_tree(tree,20)

        leaves = []
        for i in range(end_clusters.shape[0]):
            leaves.append(tree[end_clusters[i]].get_leaf_ids())

        twcp = Feature.get_occ_matrix(leaves)

        if scaled:
            maj_clust_labs = [k for k in np.argmax(twcp / np.sum(twcp, 0), 1)]
        else:
            maj_clust_labs = [k for k in np.argmax(twcp, 1)]
        matches = [twcp[i, lab] for i, lab in zip(range(len(maj_clust_labs)), maj_clust_labs)]
        match_per = np.sum(matches) / np.sum(twcp)

        flat_leaves = np.arange(np.sum([len(l) for l in leaves]))

        leaf_idxs = [(0, len(leaves[0]))]
        for i in range(1, len(leaves)):
            leaf_idxs.append((leaf_idxs[-1][1], leaf_idxs[-1][1] + len(leaves[i])))

        rand_match_pers = []
        for i in range(iterations):
            np.random.shuffle(flat_leaves)
            leaves = [flat_leaves[leaf_idxs[j][0]:leaf_idxs[j][1]] for j in range(len(leaves))]
            twcp = Feature.get_occ_matrix(leaves)
            if scaled:
                maj_clust_labs = [k for k in np.argmax(twcp / np.sum(twcp, 0), 1)]
            else:
                maj_clust_labs = [k for k in np.argmax(twcp, 1)]
            matches = [twcp[j, lab] for j, lab in zip(range(len(maj_clust_labs)), maj_clust_labs)]
            rand_match_pers.append(np.sum(matches) / np.sum(twcp))

        return ("%s,%.4f,%.4f,%.4f,%.4f" % (
        ("Scaled" if scaled else "Not scaled"), match_per,  np.average(rand_match_pers),
         match_per / np.average(rand_match_pers) - 1, np.sum(np.array(rand_match_pers) > match_per) / iterations))

    @staticmethod
    def evaluate_clustering_old(z, cc,cut=5):
        import CatClust
        tree = CatClust.Tree.Z_to_tree_dict(z, cc)
        end_clusters = Feature.get_end_cluster_ids(tree)

        if len(end_clusters) == 0:
            end_clusters = []
            root = list(tree.values())[-1]
            nodes = [root]
            new_nodes = []
            for i in range(cut):
                new_nodes = []
                for node in nodes:
                    if node.left != None:
                        new_nodes.append(node.left)
                    if node.right != None:
                        new_nodes.append(node.right)
                nodes = new_nodes

        end_clusters = [node.id for node in new_nodes]
        end_clusters = np.unique(end_clusters)

        leaves = []
        for i in range(end_clusters.shape[0]):
            leaves.append(tree[end_clusters[i]].get_leaf_ids())

        with open("true_prim_type") as f:
            true_labs = [l.rstrip("\n").lower() for l in f]
        true_labs = np.array(true_labs)

        leaf_labels = np.zeros(len(leaves), dtype=object)
        for i in range(leaf_labels.shape[0]):
            leaf_labels[i] = np.zeros(len(leaves[i]),dtype=np.dtype('<U20'))

        for i in range(len(leaves)):
            for j in range(len(leaves[i])):
                leaf_labels[i][j] = true_labs[leaves[i][j]]

        unique_leaf_labels = np.unique([k for l in leaf_labels for k in np.unique(l) if k != "unknown"])

        twcp = np.zeros((leaf_labels.shape[0], unique_leaf_labels.shape[0]))

        for i in range(len(leaves)):
            for j in range(unique_leaf_labels.shape[0]):
                occs = np.where(unique_leaf_labels[j] == np.array(leaf_labels[i]))[0].shape[0]
                twcp[i][j] += occs

        # Sum of occurance of each unique label = 1
        scaled_leaf_occ = twcp / np.sum(twcp, 0)

        maj_clust_labs = [k for k in np.argmax(scaled_leaf_occ, 1)]

        occ = scaled_leaf_occ.copy()
        #occ[:,17] = -1
        scaled_iter_maj_clust_labs = np.zeros(occ.shape[0], dtype=int)

        mask = np.ones(occ.shape[0],bool)
        for _ in range(occ.shape[0]):
            best_cluster = np.argmax(np.max(occ, 1))
            best_label = np.argmax(occ[best_cluster])
            scaled_iter_maj_clust_labs[best_cluster] = best_label
            occ[best_cluster,:] = -1
            occ[:,best_label] = -1
            mask[best_cluster] = False
            if np.max(occ) == -1:
                scaled_iter_maj_clust_labs[mask] = -1

        occ = twcp.copy()
        iter_maj_clust_labs = np.zeros(occ.shape[0], dtype=int)

        for _ in range(occ.shape[0]):
            best_cluster = np.argmax(np.max(occ, 1))
            best_label = np.argmax(occ[best_cluster])
            iter_maj_clust_labs[best_cluster] = best_label
            occ[best_cluster, :] = -1
            occ[:, best_label] = -1

        tp_fp_fn_tn = []

        #scaled_iter_maj_clust_labs = np.array(maj_clust_labs)

        for i in range(occ.shape[0]):
            if scaled_iter_maj_clust_labs[i] == -1:
                tp, fp, fn, tn = (0., 0., 0., 0.)
            else:
                tp = twcp[i,scaled_iter_maj_clust_labs[i]]
                fp = np.sum(twcp[i, :]) - tp
                fn = np.sum(twcp[:, scaled_iter_maj_clust_labs[i]]) - tp
                tn = np.sum(twcp) - tp - fp - fn
            tp_fp_fn_tn.append((tp,fp,tn,fn))

        perf = []
        for i in range(occ.shape[0]):
            tp,fp,tn,fn = tp_fp_fn_tn[i]

            if np.sum(tp_fp_fn_tn[i]) == 0:
                accuracy, precision, recall, f_score = (0.,0.,0.,0.)
            else:
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f_score = 2 * ((precision * recall) / (precision + recall))

            perf.append((accuracy, precision, recall, f_score))

        sub_cluster_labels = []
        for i in range(occ.shape[0]):
            if scaled_iter_maj_clust_labs[i] == -1:
                sub_cluster_labels.append("unknown")
            else:
                sub_cluster_labels.append(unique_leaf_labels[scaled_iter_maj_clust_labs[i]])

        cats = []
        for i in range(end_clusters.shape[0]):
            print(len(tree), end_clusters[i], tree[end_clusters[i]].cat)
            cats.append(tree[end_clusters[i]].cat[0])

        types = np.load("types.npy")
        f_used = [types[t] for t in cats]

        for i in range(len(perf)):
            print("%s&%.4f&%.4f&%.4f&%.4f\\\\" % (sub_cluster_labels[i], perf[i][0], perf[i][1], perf[i][2], perf[i][3]))

        for i in range(4):
            n = [r[i] for r in perf]
            nn = np.array(n)[np.invert(np.isnan(n))]
            print("%.4f %.4f %.4f %.4f %s" % (np.max(nn), np.median(nn), np.average(nn),np.min(nn), n))

        return sub_cluster_labels, perf



        """
#I WANT
pre, recall, f-score
 TP TN FP FN
  Label each cluster
   1. true labels of all leaves
   2. calculate scaled majority of type in each cluster
   3. - label after majority in each cluster
      - label after majority of rest in each cluster, starting with largest margins of scaled majority
   Given label, calculate TP TN FP FN for each cluster
   Supplement with true category label and compare to given label
        """



def _test_():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    # _test_()
    pass

