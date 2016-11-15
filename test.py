import clustering


def test():
    donor_fns = ["example_files/dona", "example_files/donb", "example_files/donc",
                 "example_files/dond", "example_files/done", "example_files/donf"]
    ref_fns = ["example_files/refa", "example_files/refb", "example_files/refc", "example_files/refd"]
    ref_types = ["a", "b", "c", "a"]
    #dist = "forbes-corr"
    #linkage_method = "complete"
    #feature_type = "forbes-corr"

    import CatClust
    for linkage_method in CatClust.linkage_methods.keys():
        for dist in clustering.get_legal_dist_measures():
            for feature_type in clustering.get_legal_feature_types():
                if linkage_method != "medium variance":
                    if (feature_type == "forbes-corr"  or dist == "forbes-corr") and feature_type != dist:
                        continue

                    clustering.cluster(donor_fns, ref_fns, ref_types, dist, linkage_method, feature_type,k=0.01,lock=1)





if __name__ == "__main__":
    test()

    #print(z)
    #print(cc)
    #print(tree)
