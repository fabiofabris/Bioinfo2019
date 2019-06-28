import pandas
import numpy as np


class DatasetLoader:

    def __init__(self):
        self.base_path = "../datasets/"

    """
        This method removes features with very low support or very low standard deviation
    """
    @staticmethod
    def clear_features(features, binary_features):

        if binary_features:
            s = features.sum(axis=0)

            to_remove = s[s <= 10]
            for row in to_remove.index:
                print("WARNING: Removing feature '" + row + "' due to very low <= 10 sum.")
                features.drop(row, axis=1, inplace=True)
        else:
            stds = features.std(axis=0)

            to_remove = stds[stds < 1.0e-10]
            for row in to_remove.index:
                print("WARNING: Removing feature '" + row + "' due to very low < 1.0e-10 standard deviation.")
                features.drop(row, axis=1, inplace=True)

    """
        Loads a single projection (embedding) given the feature name.
    """
    def load_projection(self, projection_fname):
        classes = pandas.read_csv(self.base_path + "class_labels.csv", sep=",", header=0, na_values=["?"])
        classes.dropna(subset=["class_Brain.Alzheimer"], inplace=True)
        class_indexes = classes.columns[1:]
        classes = classes.set_index("entrezId")

        features = pandas.read_csv(projection_fname, sep=",", header=0, na_values=["?"])
        features = features.set_index("entrezId")

        dataset = features.join(classes, how='right', lsuffix="_lsuf")
        dataset.class_indexes = class_indexes

        return dataset

    """
        Adds a new feature type to the dataset.
    """
    def add_feature_type(self, dataset, feature_type, binary_features=False):
        feature = self.load_features(feature_type, binary_features)
        feature_indexes = dataset.feature_indexes
        feature_indexes = feature_indexes.append(feature.columns[1:])

        class_indexes = dataset.class_indexes

        dataset = dataset.join(feature, how='outer', lsuffix="_lsuf")
        dataset.dropna(subset=["class_Brain.Alzheimer"], inplace=True)

        dataset.feature_indexes = feature_indexes
        dataset.class_indexes = class_indexes
        return dataset


    """
        Loads the features.
    """
    def load_features(self, feature_type, binary_features):
        if binary_features:
            f = open(self.base_path + feature_type)

            # assuming that the last column always contains the totals
            feat_names = f.readline().strip().split(",")[1:-1]
            f.close()
            dtypes = {}
            for name in feat_names:
                dtypes[name] = bool

            features = pandas.read_csv(self.base_path + feature_type, sep=",", header=0, na_values=["?"], dtype=dtypes)
        else:
            features = pandas.read_csv(self.base_path + feature_type, sep=",", header=0, na_values=["?"])

        features = features.set_index("entrezId")
        DatasetLoader.clear_features(features, binary_features)

        return features

    """
        Loads the dataset given a feature type.
        This involves loading the file with the features, loading the file with the classes and combining them.
    """
    def load_dataset(self, feature_type, binary_features=False):

        classes = pandas.read_csv(self.base_path + "class_labels.csv", sep=",", header=0, na_values=["?"])
        classes.dropna(subset=["class_Brain.Alzheimer"], inplace=True)
        class_indexes = classes.columns[1:]
        classes = classes.set_index("entrezId")
        features = self.load_features(feature_type, binary_features)

        # remove features with low standard deviation

        feature_indexes = features.columns[1:]

        dataset = features.join(classes, how='right', lsuffix="_lsuf")

        del features
        del classes

        # dataset.drop("entrezId_lsuf", axis=1, inplace=True)

        # dataset = dataset.set_index("entrezId")
        dataset.feature_indexes = feature_indexes
        dataset.class_indexes = class_indexes
        dataset.binary_features = binary_features

        return dataset

