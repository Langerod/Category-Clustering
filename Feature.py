__author__ = 'krilange'
import numpy as np
from collections import defaultdict as dd


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


def _test_():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    # _test_()
    pass

