from keras.layers import Dense
from keras.models import Model
from keras.layers import Dropout
from keras.callbacks import CSVLogger
from keras.callbacks import Callback
from keras.optimizers import Adam
import pandas

from keras.layers import Input
from keras.models import load_model

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pickle

from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.multiclass import OneVsRestClassifier

import numpy as np


# dummy class
class DeepNet:
    pass

"""
    Helper class to "split" the dataset into features and classes
"""
class PrepareDatasetDiseaseHie:

    def __init__(self):
        self.feature_type_indexes = None

    @staticmethod
    def prepare_dataset(dataset):

        class_indexes = dataset.class_indexes
        classes = dataset[class_indexes]

        features = dataset.drop(labels=class_indexes, axis=1)

        # features.drop(columns=["entrezId"], inplace=True)

        return features, classes

"""
    Helper class to store the evolution of the auroc.
"""
class Losses(Callback):
    def __init__(self, my_model, train_p, test_p, lc_file_name=None):
        Callback.__init__(self)
        self.my_model = my_model
        self.test = test_p
        self.train = train_p
        self.lc_file = None
        self.lc_file_name = lc_file_name

    def on_train_begin(self, logs=None):
        if self.lc_file_name is not None:
            self.lc_file = open(self.lc_file_name, "w")

    def on_train_end(self, logs=None):
        self.evaluate(-1)
        if self.lc_file_name is not None:
            self.lc_file.close()

    def evaluate(self, epoch):
        print("begin callback evaluate")
        ids, preds, classes = self.my_model.evaluate(self.train)
        auroc_train = metrics.roc_auc_score(classes, preds, average="weighted")
        log_train = metrics.log_loss(classes, preds)

        if self.test is not None:
            ids, preds, classes = self.my_model.evaluate(self.test)
            auroc_test = metrics.roc_auc_score(classes, preds, average="weighted")
            log_test = metrics.log_loss(classes, preds)

            if self.lc_file is not None:
                self.lc_file.write(str(epoch) + "," + str(auroc_train) + "," + str(auroc_test) + "," + str(log_train) +
                                   "," + str(log_test) + "\n")
                self.lc_file.flush()
            else:
                print(str(epoch) + "," + str(auroc_train) + "," + str(auroc_test) + "," + str(log_train) + "," +
                      str(log_test))
        else:
            if self.lc_file is not None:
                self.lc_file.write(str(epoch) + "," + str(auroc_train) + "," + str(log_train) + "\n")
                self.lc_file.flush()
            else:
                print(str(epoch) + "," + str(auroc_train) + "," + str(log_train))
        print("end callback evaluate")

    def on_epoch_end(self, epoch, logs=None):

        if epoch % 10 is not 0:
            return
        self.evaluate(epoch)


"""
    A multilabel light gbm classifier.
"""
class BoostedTreeModel(PrepareDatasetDiseaseHie):

    def __init__(self):
        PrepareDatasetDiseaseHie.__init__(self)
        self.model = None
        self.priors = None
        self.params = {}

    def train(self, train_p, test_p=None, lc_file=None):
        features, classes = PrepareDatasetDiseaseHie.prepare_dataset(train_p)
        self.priors = classes.mean(axis=0)

        clf_multilabel = OneVsRestClassifier(lgb.LGBMClassifier())

        clf_multilabel.fit(features, classes)
        self.model = clf_multilabel

    def evaluate(self, test):
        features, classes = PrepareDatasetDiseaseHie.prepare_dataset(test)
        predictions = pandas.DataFrame(self.model.predict_proba(features))
        ids = test.index.tolist()

        return ids, predictions, classes

    def save_model(self, f_name):
        f = open(f_name + "_model", "w")
        f.close()


"""
    A coustom normlizer that ignores binary features.
"""
class CustomScaler:

    def __init__(self):
        self.means={}
        self.vars={}

    def fit(self, x):
        # for each column
        for col in x.columns:

            # get type of first element
            t = type(x.loc[:,col].iloc[0])

            if t != np.float64:
                # if the column is binary, do nothing
                pass
            else:
                # if the column is numeric, get the mean and standard deviation
                self.means[col] = x[col].mean()
                self.vars[col] = x[col].std()

    def transform(self, x):
        # for each column
        for col in x.columns:
            # if the column is binary, do nothing
            t = type(x.loc[:,col].iloc[0])
            if t != np.float64:
                continue
            else:
                x[col] = (x[col] - self.means[col])/(self.vars[col])
        return x


"""
    The modular DNN model, trains a DNN using a single feature type.
"""
class ModularModel(PrepareDatasetDiseaseHie, DeepNet):

    def __init__(self, epochs):
        PrepareDatasetDiseaseHie.__init__(self)
        self.epochs = epochs
        self.scaler = None
        self.model = None
        self.priors = None
        self.projection_layer = None

    @staticmethod
    def load(net_file, priors_file, scaler_file):
        modular_model = ModularModel(-1)

        modular_model.model = load_model(net_file)

        with open(priors_file, "rb") as f:
            modular_model.priors = pickle.load(f)

        with open(scaler_file, "rb") as f:
            modular_model.scaler = pickle.load(f)

        return modular_model

    """
        Gets the projections (embeddings) for the dataset test_p.
        If "for_class" is provided, calculate the embedding for a specific class.
    """
    def get_projection(self, test_p, for_class = None):

        class_indexes = test_p.class_indexes
        test_p = test_p.dropna(axis=0)
        test_p.class_indexes = class_indexes

        features, classes = PrepareDatasetDiseaseHie.prepare_dataset(test_p)
        ids = test_p.index.tolist()

        prepared_dataset = self.scaler.transform(features)
        predictions = self.projection_layer.predict(prepared_dataset)

        if for_class != None:
            w = self.model.layers[-1].get_weights()[0]

            class_index = class_indexes.get_loc(for_class)
            wei = w[:,class_index]
            predictions = np.multiply(predictions, wei)

            return ids, predictions, classes
        else:
            return ids, predictions, classes

    @staticmethod
    def remove_na(features, classes):
        to_remove = []
        # for each row
        for index, row in features.iterrows():
            # if there is any NA feature
            if row.isna().any():
                to_remove.append(index)

        features.drop(inplace=True, axis=0, index=to_remove)
        classes.drop(inplace=True, axis=0, index=to_remove)

        # remove the row from the features and classes

    """
        Trains the DNN models.
    """
    def train(self, train_p, test_p=None, lc_file=None):

        print("Train shape before removing NA", train_p.shape)

        if test_p is not None:
            print("Test shape before removing NA", test_p.shape)

        # prepares the dataset
        features, classes = PrepareDatasetDiseaseHie.prepare_dataset(train_p)
        ModularModel.remove_na(features, classes)

        self.scaler = CustomScaler()
        self.projection_layer = None
        self.priors = classes.mean(axis=0)

        # builds the DNN
        input_layer = Input(shape=(len(features.columns),), name='main_input')

        x = Dense(64, kernel_initializer='uniform', activation='relu')(input_layer)
        x = Dropout(0.5)(x)
        x = Dense(32, kernel_initializer='uniform', activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(16, kernel_initializer='uniform', activation='relu')(x)
        self.projection_layer = Model(input_layer, x)
        x = Dropout(0.5)(x)

        self.scaler.fit(features)
        features = self.scaler.transform(features)

        output = Dense(len(classes.columns), kernel_initializer='uniform', activation='relu', name="output")(x)
        model = Model(inputs=input_layer, outputs=output)

        opt = Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
        self.model = model
        print(model.summary())

        # if test_p is provided, prepare it too.
        if test_p is not None:
            test_f, test_c = PrepareDatasetDiseaseHie.prepare_dataset(test_p)
            ModularModel.remove_na(test_f, test_c)
            test_f = self.scaler.transform(test_f)
        else:
            test_f = None
            test_c = None

        print("Train shape after removing NA", train_p.shape)

        if test_p is not None:
            print("Test shape after removing NA", test_p.shape)

        # train the model
        if test_p is not None:
            model.fit(features, classes, epochs=self.epochs, batch_size=512*2,  verbose=0,
                      callbacks=[CSVLogger(lc_file + "_noprior")], validation_data=(test_f, test_c))
        else:
            model.fit(features, classes, epochs=self.epochs, batch_size=512*2,  verbose=0,
                      callbacks=[CSVLogger(lc_file + "_noprior")])

    def save_model(self, f_name):
        self.model.save(f_name + "_model")

        with open(f_name + "_priors", "wb") as f:
            pickle.dump(self.priors, f)

        with open(f_name + "_scaler", "wb") as f:
            pickle.dump(self.scaler, f)

    """
        Prepares the testing dataset, substituting na values with 0. 
    """
    def fill_prior(self, test_p):
        features, classes = PrepareDatasetDiseaseHie.prepare_dataset(test_p)

        is_na = []
        i = 0
        for index, row in features.iterrows():
            if row.isna().any():
                is_na.append(i)
            i += 1

        features.fillna(value=0, inplace=True)
        features = self.scaler.transform(features)
        return is_na, features, classes

    """
        Gets the predictions for test_p
    """
    def evaluate(self, test_p):

        is_na, features, classes = self.fill_prior(test_p)
        print("******* number of instances with na values", len(is_na))

        predictions = pandas.DataFrame(self.model.predict(features))

        # substitute the predictions for missing instances with the prior.
        for i in is_na:
            predictions.iloc[i, :] = self.priors.values

        ids = test_p.index.tolist()

        return ids, predictions, classes


"""
    Trains the model joining feature types and trainign the last layer.
"""
class JoinedModel(PrepareDatasetDiseaseHie, DeepNet):

    def __init__(self, epochs):
        PrepareDatasetDiseaseHie.__init__(self)
        self.epochs = epochs
        self.scaler = None
        self.model = None
        self.priors = None
        self.projection_layer = None

    def get_projection(self, test_p, for_class=None):

        class_indexes = test_p.class_indexes
        test_p = test_p.dropna(axis=0)
        test_p.class_indexes = class_indexes

        features, classes = PrepareDatasetDiseaseHie.prepare_dataset(test_p)
        ids = test_p.index.tolist()

        prepared_dataset = self.scaler.transform(features)
        predictions = self.projection_layer.predict(prepared_dataset)


        if for_class != None:
            w = self.model.layers[-1].get_weights()[0]

            class_index = class_indexes.get_loc(for_class)
            wei = w[:,class_index]
            predictions = np.multiply(predictions, wei)

            return ids, predictions, classes
        else:
            return ids, predictions, classes


    """
        Adds 2 layers to the end of the DNN and trains it to combine the modules.
    """
    def train(self, train_p, test_p=None, lc_file=None):

        features, classes = PrepareDatasetDiseaseHie.prepare_dataset(train_p)

        self.scaler = None
        self.projection_layer = None
        self.priors = classes.mean(axis=0)

        input_layer = Input(shape=(len(features.columns),), name='main_input')

        x = Dense(32, kernel_initializer='uniform', activation='relu')(input_layer)
        x = Dropout(0.5)(x)
        x = Dense(16, kernel_initializer='uniform', activation='relu')(x)
        self.projection_layer = Model(input_layer, x)
        x = Dropout(0.5)(x)

        self.scaler = StandardScaler()
        self.scaler.fit(features)

        test_f, test_c = PrepareDatasetDiseaseHie.prepare_dataset(test_p)
        test_f = self.scaler.transform(test_f)

        output = Dense(len(classes.columns), kernel_initializer='uniform', activation='relu', name="output")(x)
        model = Model(inputs=input_layer, outputs=output)
        print(model.summary())

        opt = Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
        self.model = model

        features = self.scaler.transform(features)
        model.fit(features, classes, epochs=self.epochs, batch_size=512*4,  verbose=0,
                  callbacks=[CSVLogger(lc_file + "_noprior")], validation_data=(test_f, test_c))

    """
        Gets the predictions of the model.
    """
    def evaluate(self, test_p):

        features, classes = PrepareDatasetDiseaseHie.prepare_dataset(test_p)
        features = self.scaler.transform(features)

        predictions = pandas.DataFrame(self.model.predict(features))

        ids = test_p.index.tolist()

        return ids, predictions, classes
