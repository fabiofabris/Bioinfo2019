from numpy.random import seed
from tensorflow import set_random_seed
import random

# sets global random seeds
random.seed(1)
seed(1)
set_random_seed(1)

from sklearn import metrics
import pandas
import train_model
import cross_validation
from cross_validation import CrossValidation
import load_dataset
import sys
import os

import logging
logging.basicConfig(filename='/tmp/deep.log', level=logging.DEBUG)


# deprecated
def set_seeds(s):
    pass
    # seed(s)
    # set_random_seed(s)


def create_dir_structure(base_folder):
    def make_dir(newpath):
        if not os.path.exists(newpath):
            os.makedirs(newpath)

    make_dir(base_folder)
    make_dir(base_folder + "/predictions")
    make_dir(base_folder + "/predictions_train")
    make_dir(base_folder + "/projections")
    make_dir(base_folder + "/lcs")
    make_dir(base_folder + "/folds")
    make_dir(base_folder + "/models")


def file_exists(f_name):
    return os.path.isfile(f_name)


"""
    Runs one fold of the 10-cv or builds a model with the whole data (if fold == -10)
"""
def run_cv(s, fold, base_folder, params, dataset, classifier_builder):

    if file_exists(base_folder + "/models/model_" + str(fold) + "_model"):
        print("will not run")
        return

    if fold == -1:
        if file_exists(base_folder + "/models/model_final_net"):
            print("will not run")
            return

    # set random seeds
    set_seeds(s)

    create_dir_structure(base_folder)

    # build the final model
    if fold == -1:
        classifier = classifier_builder(**params)

        classifier.train(dataset, lc_file=base_folder + "/lcs/lc_fold_-1")
        ids, predictions, classes_d = classifier.evaluate(dataset)

        CrossValidation.write_results(ids, predictions, classes_d, -1, base_folder + "/predictions/")
        classifier.save_model(base_folder + "/models/model_final")

        ids, projection, classes = classifier.get_projection(dataset)
        cross_validation.save_projections(projection, base_folder + "/projections/fold_train_" + str(fold), ids)

        for class_name in classes_d.columns:
            ids, projection, classes = classifier.get_projection(dataset, class_name)
            cross_validation.save_projections(projection,
                             base_folder + "/projections/class_fold_train_" + class_name + "_" + str(fold), ids)

    else:
        # run 1 fold of 10 cv, saving each model
        cross_valid = CrossValidation(dataset, classifier_builder, params, base_folder)
        cross_valid.run(fold)

"""
    Gets the entrez ids of the training and testing instances for each fold
"""
def read_entrez_indexes(fold_file):
    with open(fold_file) as f:
        train_indexes = eval(f.readline())
        test_indexes = eval(f.readline())
    return train_indexes, test_indexes


"""
    Joins the projections "embeddings" to create a combined dataset.
"""
def merge_projections(projection_file_list, fold_entrez):

    classes = pandas.read_csv("../datasets/class_labels.csv",
                              sep=",", header=0, na_values=["?"])
    classes = classes.dropna(subset=["class_Brain.Alzheimer"])
    classes = classes.set_index("entrezId")

    class_indexes = classes.columns[:]

    combined = None

    i = 0
    for projection_file in projection_file_list:

        if combined is None:
            combined = pandas.read_csv(projection_file, sep=",", header=0, na_values=["?"])
            combined = combined.set_index("entrezId")
            continue

        projection = pandas.read_csv(projection_file, sep=",", header=0, na_values=["?"])
        projection = projection.set_index("entrezId")
        combined = projection.join(combined, how="outer", rsuffix="_" + str(i))

        i += 1

    combined = combined.join(classes, how='right', lsuffix="_lsuf")

    if fold_entrez is not None:
        combined = combined.loc[fold_entrez]

    combined.fillna(-1, inplace=True)
    combined.class_indexes = class_indexes

    return combined


"""
    Helper method to evaluate the performance of the modular DNN with a given list of modules using an internal validation set.
"""
def evaluate_joined_models_seq(base_out_folder, base_in_folder, models_to_load, cur_fold, epochs, s):

    # set random seeds
    set_seeds(s)

    # load the projection testing and training entrez ids for this fold
    training_entrez, testing_entrez = read_entrez_indexes(base_in_folder + models_to_load[0] + "/folds/fold_" +
                                                          str(cur_fold))

    # for each projection
    training_projections = [base_in_folder + model_name + "/projections/fold_train_" + str(cur_fold) +
                            "_projection.csv" for model_name in models_to_load]
    training_datasets = merge_projections(training_projections, training_entrez)

    class_indexes = training_datasets.class_indexes
    training_datasets  = training_datasets.sample(frac=1).reset_index(drop=True)
    training_datasets.class_indexes = class_indexes

    internal_training_datasets = training_datasets.iloc[:int(training_datasets.shape[0]*0.7)]
    internal_training_datasets.class_indexes = training_datasets.class_indexes

    internal_testing_datasets = training_datasets.iloc[int(training_datasets.shape[0]*0.7):]
    internal_testing_datasets.class_indexes = training_datasets.class_indexes 

    model = train_model.JoinedModel(epochs)
    model.train(internal_training_datasets, internal_testing_datasets, 
        lc_file=base_out_folder + "lcs/internal_lc_fold_" + str(cur_fold))

    ids, predictions, classes = model.evaluate(internal_testing_datasets)

    auroc = metrics.roc_auc_score(classes, predictions, average="weighted")

    return auroc


"""
    Runs one fold of the 10-cv procedure for the joined models.
"""
def run_joined_models(base_out_folder, base_in_folder, models_to_load, cur_fold, epochs, s):

    create_dir_structure(base_out_folder)

    # set random seeds
    set_seeds(s)

    # load the projection testing and training entrez ids for this fold. If the results alread exist, do nothing.
    if cur_fold != -1:
        f_to_check = base_out_folder + "/predictions/predictions_fold_" + str(cur_fold)
        if file_exists(f_to_check):
            print("not running")
            print("file exists", f_to_check)
            return

        training_entrez, testing_entrez = read_entrez_indexes(base_in_folder + models_to_load[0] + "/folds/fold_" +
                                                              str(cur_fold))

        # gets the projection
        training_projections = [base_in_folder + model_name + "/projections/fold_train_" + str(cur_fold) +
                                "_projection.csv" for model_name in models_to_load]
        training_datasets = merge_projections(training_projections, training_entrez)

        testing_projections = [base_in_folder + model_name + "/projections/fold_test_" + str(cur_fold) +
                               "_projection.csv" for model_name in models_to_load]
        testing_datasets = merge_projections(testing_projections, testing_entrez)

        # Trains the model and gets the predictions.
        model = train_model.JoinedModel(epochs)
        model.train(training_datasets, testing_datasets, lc_file=base_out_folder + "lcs/lc_fold_" + str(cur_fold))
        ids, predictions, classes = model.evaluate(testing_datasets)

        # Writes the results.
        CrossValidation.write_results(ids, predictions, classes, cur_fold, base_out_folder + "/predictions/")

    # Runs the training algorithm to the whole dataset
    else:
        if file_exists(base_out_folder + "/projections/_projection.csv"):
            print("not running")
            return

        training_projections = [base_in_folder + model_name + "/projections/fold_train_-1_projection.csv" for model_name in
                                models_to_load]
        training_datasets = merge_projections(training_projections, None)
        model = train_model.JoinedModel(epochs)
        model.train(training_datasets, training_datasets, lc_file=base_out_folder + "lcs/lc_fold_" + str(cur_fold))
        ids, projection, classes = model.get_projection(training_datasets)
        cross_validation.save_projections(projection, base_out_folder + "/projections/", ids)

        for class_name in classes.columns:
            ids, projection, classes = model.get_projection(training_datasets, class_name)
            cross_validation.save_projections(projection,
                base_out_folder + "/projections/class_fold_train_" + class_name + "_-1", ids)

        ids, predictions, classes = model.evaluate(training_datasets)
        CrossValidation.write_results(ids, predictions, classes, cur_fold, base_out_folder + "/predictions/")


"""
    Runs the per-feature type algorithms.
"""
def run_base_models(base_folder, params, seed_n, cur_fold, dataset, classifier_builder=train_model.ModularModel):
    run_cv(seed_n, cur_fold, base_folder, params, dataset, classifier_builder)


"""
    Reads the predictions for the stacking approach.
"""
def read_predictions(f_name, f_id):
    all_probs = []
    with open(f_name) as f:
        lines = f.readlines()
        column_names = ["entrezId"]

        for predictions in lines[1:]:
            predictions  = predictions .strip()
            predictions_s = predictions.split(",")
            entrez = int(predictions_s[1])

            data = [entrez]
            tmp_names = []
            for prediction in predictions_s[3:]:
                class_label = prediction.split("/")[0]
                probability = float(prediction.split("/")[1])
                data.append(probability)
                if len(column_names) == 1:
                    tmp_names.append(f_id + "_" + class_label)
            all_probs.append(data)

            if len(column_names) == 1:
                column_names.extend(tmp_names)

    df = pandas.DataFrame(data=all_probs,  columns=column_names)
    df.set_index("entrezId", inplace=True)

    return df


def merge_predictions(prediction_files, f_ids):
    predictions = read_predictions(prediction_files[0], f_ids[0])
    for i, f_name in enumerate(prediction_files[1:]):
        cur_predictions = read_predictions(f_name, f_ids[i+1])
        predictions = predictions.join(cur_predictions, how='right')
    return predictions


"""
    Runs the boosted tree stacking procedure.
"""
def run_joined_models_tree(base_out_folder, base_in_folder, models_to_load, cur_fold, s):

    create_dir_structure(base_out_folder)

    # set random seeds
    set_seeds(s)

    # checks if result file already exists. if so, do not run the algorithm.
    if cur_fold != -1:
        f_to_check = base_out_folder + "/predictions/predictions_fold_" + str(cur_fold)
        if file_exists(f_to_check):
            print("not running")
            print("file exists", f_to_check)
            return

        # load the testing and training entrez ids for this fold
        training_entrez, testing_entrez = read_entrez_indexes(base_in_folder + models_to_load[0] + "/folds/fold_" +
                                                              str(cur_fold))

        testing_predictions = [base_in_folder + model_name + "/predictions/predictions_fold_" + str(cur_fold)
                                    for model_name in models_to_load]
        testing_dataset = merge_predictions(testing_predictions, models_to_load)

        training_predictions = [base_in_folder + model_name + "/predictions_train/predictions_fold_" + str(cur_fold)
                               for model_name in models_to_load]

        training_dataset = merge_predictions(training_predictions, models_to_load)

        classes = pandas.read_csv(load_dataset.DatasetLoader.base_path + "class_labels.csv", sep=",", header=0, na_values=["?"])
        classes.dropna(subset=["class_Brain.Alzheimer"], inplace=True)
        class_indexes = classes.columns[1:]
        classes.set_index("entrezId", inplace=True)

        training_dataset = training_dataset.join(classes)
        training_dataset.class_indexes = class_indexes

        testing_dataset = testing_dataset.join(classes)
        testing_dataset.class_indexes = class_indexes

        model = train_model.BoostedTreeModel()
        model.train(training_dataset, testing_dataset)
        ids, predictions, classes = model.evaluate(testing_dataset)

        CrossValidation.write_results(ids, predictions, classes, cur_fold, base_out_folder + "/predictions/")
    else:

        if file_exists(base_out_folder + "/projections/_projection.csv"):
            print("not running")
            return

        training_projections = [base_in_folder + model_name + "/projections/fold_train_-1_projection.csv" for model_name in
                                models_to_load]
        training_datasets = merge_projections(training_projections, None)
        model = train_model.JoinedModel(epochs)
        model.train(training_datasets, training_datasets, lc_file=base_out_folder + "lcs/lc_fold_" + str(cur_fold))
        ids, projection, classes = model.get_projection(training_datasets)
        cross_validation.save_projections(projection, base_out_folder + "/projections/", ids)


"""
    Main method.
    Check the run.sh script for the list of parameters.
"""
if __name__ == "__main__":
    params_sys = eval(sys.argv[1])

    if params_sys["base_model"] == "module":
        dl = load_dataset.DatasetLoader()
        params = {"epochs": params_sys["epochs"]}
        dataset = dl.load_dataset(params_sys["feature_type"], binary_features=params_sys["binary_feature"] == "True")
        run_base_models(params_sys["base_folder"], params, params_sys["seed"], params_sys["fold"],
                        dataset)

    if params_sys["base_model"] == "joined_seq":

        create_dir_structure(params_sys["base_folder"])
        selected_modules = []
        available_modules = list(params_sys["models_to_load"])
        print("Available models:", str(available_modules))

        has_improved = True
        best_auroc = 0
    
        while(len(available_modules) > 0 and has_improved):

            has_improved = False
            for module in available_modules:

                selected_modules.append(module)
                auroc = evaluate_joined_models_seq(params_sys["base_folder"], params_sys["base_in_folder"], 
                        selected_modules, params_sys["fold"], params_sys["epochs"], params_sys["seed"])

                print("tested %s, AUROC: %f:" % (str(selected_modules), auroc))

                selected_modules.pop()

                if auroc > best_auroc:
                    best_auroc = auroc
                    selected_module_so_far = module
                    has_improved = True
                    print("Found best add: %s, AUROC: %f:" % (selected_module_so_far, auroc))

            if has_improved:
                selected_modules.append(selected_module_so_far)
                available_modules.remove(selected_module_so_far)
                print("Adding final: %s" % str(selected_modules))
                print("Available: %s" % str(available_modules))

        print("Run final model: %s." % (str(selected_modules)))
        run_joined_models(params_sys["base_folder"], params_sys["base_in_folder"], 
                selected_modules, params_sys["fold"], params_sys["epochs"], params_sys["seed"])

    if params_sys["base_model"] == "joined":
        run_joined_models(params_sys["base_folder"], params_sys["base_in_folder"], params_sys["models_to_load"], params_sys["fold"], params_sys["epochs"], params_sys["seed"])

    if params_sys["base_model"] == "full_dataset":
        params = {"epochs": params_sys["epochs"]}
        dl = load_dataset.DatasetLoader()
        ds = dl.load_dataset("base_expression_gtex_features.csv", binary_features=False)
        ds = dl.add_feature_type(ds, "pathdipall_features.csv", binary_features=True)
        ds = dl.add_feature_type(ds, "go_features.csv", binary_features=True)
        ds = dl.add_feature_type(ds, "ppi_features.csv", binary_features=True)
        ds.binary_features = None
        ds.fillna(-1, inplace=True)        

        run_base_models(params_sys["base_folder"], params, params_sys["seed"], params_sys["fold"], ds)

    if params_sys["base_model"] == "boosted_tree":
        params = {}
        dl = load_dataset.DatasetLoader()
        dataset = dl.load_dataset(params_sys["feature_type"], binary_features=params_sys["binary_feature"] == "True")
        run_base_models(params_sys["base_folder"], params, params_sys["seed"], params_sys["fold"],
                        dataset, classifier_builder=train_model.BoostedTreeModel)

    if params_sys["base_model"] == "joined_tree":
        run_joined_models_tree(params_sys["base_folder"], params_sys["base_in_folder"], params_sys["models_to_load"],
                          params_sys["fold"], params_sys["seed"])

    if params_sys["base_model"] == "full_dataset_tree":
        params = {}

        dl = load_dataset.DatasetLoader()
        ds = dl.load_dataset("base_expression_gtex_features.csv", binary_features=False)
        ds = dl.add_feature_type(ds, "pathdipall_features.csv", binary_features=False)
        ds = dl.add_feature_type(ds, "go_features.csv", binary_features=False)
        ds = dl.add_feature_type(ds, "ppi_features.csv", binary_features=False)
        ds.binary_features = None
        ds.fillna(-1, inplace=True)

        run_base_models(params_sys["base_folder"], params, params_sys["seed"], params_sys["fold"], ds,
                        classifier_builder=train_model.BoostedTreeModel)

