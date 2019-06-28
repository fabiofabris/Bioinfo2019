from sklearn import metrics
from sklearn.model_selection import KFold
from train_model import DeepNet
import pandas


"""
    Saves the embeddings to a file.
"""
def save_projections(encoded, base_name, ids):
    f_name = base_name + "_projection.csv"

    encoded = pandas.DataFrame(encoded)
    encoded.insert(0, "entrezId", ids)
    encoded["entrezId"] = pandas.to_numeric(encoded["entrezId"], downcast="integer")
    encoded = encoded.set_index("entrezId")

    print(f_name)
    encoded.to_csv(f_name, index=True)


class CrossValidation:

    def __init__(self, dataset, classifier_builder, classifier_params, base_folder):
        self.dataset = dataset
        self.classifier_builder = classifier_builder
        self.classifier_params = classifier_params
        self.base_folder = base_folder

    @staticmethod
    def copy_properties(ds, train, test):
        train.binary_features = ds.binary_features
        train.class_indexes = ds.class_indexes
        train.feature_indexes = ds.feature_indexes

        test.binary_features = ds.binary_features
        test.class_indexes = ds.class_indexes
        test.feature_indexes = ds.feature_indexes

    @staticmethod
    def write_results(ids, predictions, classes, fold_id, fold_file_path):

        auroc = metrics.roc_auc_score(classes, predictions, average="weighted")

        with open(fold_file_path + "predictions_fold_" + str(fold_id), "w") as f:
            f.write("AUROC," + str(auroc) + "\n")
            for i in range(predictions.shape[0]):  # for each instance
                id_instance = ids[i]
                f.write("PREDICTION," + str(id_instance) + "," + str(fold_id) + ",")

                pred = predictions.iloc[i] # predicted class probability
                clas = classes.iloc[i] # actual class

                for j in range(clas.shape[0]):  # for each class
                    class_name = clas.index[j]
                    f.write(class_name + "/" + str(pred[j]) + "/" + str(clas[j]))
                    if j != clas.shape[0] - 1:
                        f.write(",")

                f.write("\n")


    """
        Writes the indexes of the instances used in each fold in a file.
    """
    @staticmethod
    def write_folds(train, test, f_name):
        with open(f_name, "w") as f:
            f.write(str(list(train.index.tolist())) + "\n")
            f.write(str(list(test.index.tolist())) + "\n")

    """
        Does the actual cross-validation.
        If fold_to_run is provided, will run the 10-cv for that fold only.
    """
    def run(self, fold_to_run=None):

        kf = KFold(n_splits=10, shuffle=True)

        cur_fold = 0
        # for each fold
        for train_index, test_index in kf.split(self.dataset):

            if fold_to_run is not None and fold_to_run != cur_fold:
                cur_fold += 1
                continue

            # gets the training and testing sets
            train = self.dataset.iloc[train_index, :]
            test = self.dataset.iloc[test_index, :]

            # copies the speciall properties in self.dataset to train and test. Writes the ids of the instances in a file.
            CrossValidation.copy_properties(self.dataset, train, test)
            CrossValidation.write_folds(train, test, self.base_folder + "/folds/fold_" + str(cur_fold))

            # induces the classifier
            cla = self.classifier_builder(**self.classifier_params)
            cla.train(train, test, lc_file=self.base_folder + "/lcs/lc_fold_" + str(cur_fold))

            # gets the predictions (for the testing set and the training set)
            ids_test, predictions_test, classes_test = cla.evaluate(test)
            ids_train, predictions_train, classes_train = cla.evaluate(train)

            # writes the auroc and predictions in a file.
            CrossValidation.write_results(ids_test, predictions_test, classes_test, cur_fold, self.base_folder + "/predictions/")
            CrossValidation.write_results(ids_train, predictions_train, classes_train, cur_fold, self.base_folder + "/predictions_train/")

            # do extra stuff if the classifier is a DNN
            if isinstance(cla, DeepNet):
                # gets and saves the global embedings (regardless of the classes)
                ids, projection, classes = cla.get_projection(train)
                save_projections(projection, self.base_folder + "/projections/fold_train_" + str(cur_fold), ids)

                # gets the per-class projections (by multiplying the output with the weights in the last layer)
                for class_name in  classes_train.columns:
                    ids, projection, classes = cla.get_projection(train, class_name)
                    save_projections(projection, self.base_folder + "/projections/class_fold_train_" + class_name + "_" + str(cur_fold), ids)


                ids, projection, classes = cla.get_projection(test)
                save_projections(projection, self.base_folder + "/projections/fold_test_" + str(cur_fold), ids)

                for class_name in  classes_train.columns:
                    ids, projection, classes = cla.get_projection(test, class_name)
                    save_projections(projection, self.base_folder + "/projections/class_fold_test_" + class_name + "_" + str(cur_fold), ids)

            # saves the classification model
            cla.save_model(self.base_folder + "/models/model_" + str(cur_fold))

            cur_fold += 1


