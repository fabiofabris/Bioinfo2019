# source ../../sourceMe

epochs=150
base_model="'module'"

# This scritp executes all experiments, each call to "experiment.py" will run
# one fold of the 10-cv procedure

# The paameters are as follows:
# Parameter "base_model" - The type of classification algorithm options are:
# Possible values:
#   module: Trains a DNN using a single feature type (GO, PPI, Gtex, PathDIP).
#   Parts of the resulting module (saved in the filesystem) will be used by the
#   "joined" and "joined_seq" opitions.
#   This corresponds to the rows GO, PPI, PathDIP and GTex in Table 2 of the paper.
#
#   joined: Joins a collection of pre-trained modules (GO, PPI, Gtex, PathDIP),
#   training a final layer to combine them. This corresponds to the row "Modular approach"
#   in Table 2 of the paper.
#
#   joined_seq: Similar to the previous opition, but also runs an algorithm for
#   sequentially adding feature types modules (GO, PPI, Gtex, PathDIP) to
#   select a good module combination, analogous to a forward feature selection
#   method.
#   This corresponds to the row "FSS" in Table 2 of the paper.                 
#
#   full_dataset: trains the model using the full (concatenated) dataset.
#   This corresponds to the row "Concat. all feats." in Table 2 of the paper.                 
#
#   boosted_tree: trains a boosted tree model for a given feature type (GO, PPI, Gtex, PathDIP)
#   This corresponds to the rows GO, PPI, PathDIP and GTex in Table 2 of the paper.
#
#   full_dataset_tree: trains a boosted tree model using a dataset with all
#   feature types (the concatenated dataset).
#   This corresponds to the row "Concat. all feats." in Table 2 of the paper.                 
#
#   Joined tree: Loads the pre-trained individual boosted trees (trained using the boosted_tree option) 
#   gets the predictions and trains another tree to combine them and output the final prediction (like a staking approach).
#   This corresponds to the row "Stack" in Table 2 of the paper.                 


# Parameter base_folder: where the trained models and other results will be saved.

# Parameter feature_type: for "module" and "boosted tree" opitions indicates which dataset to load
# to train the machine learning algorithms.

# Parameter "binary_feature" indicates if the features are binary (should be
# true for GO, PPI and PathDIP features).

# Parameter seed: the random seed to be used.

# Parameter fold: the fold (of the 10-fold cross validation procudure) to run. Use -1 to
# use the whole dataset for training.

# Parameter epochs: The epochs used during the DNN training.



base_folder="'results/kegg'"

feature_type="'kegg_features.csv'"
binary_feature="True"

  for fold in $(seq 0 9); do
      echo python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':${fold}, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
      python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':${fold}, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
  done
  
  echo python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':-1, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
  python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':-1, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
 
   
base_folder="'results/gtex'"
feature_type="'base_expression_gtex_features.csv'"
binary_feature="False"
for fold in $(seq 0 9); do
    echo python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':${fold}, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
    python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':${fold}, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
done

echo python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':-1, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':-1, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"


base_folder="'/home/fabiofabris/results_proj/pathdip'"
feature_type="'pathdipall_features.csv'"
binary_feature="True"
  for fold in $(seq 0 9); do
      echo python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':${fold}, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
      python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':${fold}, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
  done

echo python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':-1, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':-1, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"

base_folder="'/home/fabiofabris/results_proj/ppi'"
feature_type="'ppi_features.csv'"
binary_feature="True"
  for fold in $(seq 0 9); do
      echo python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':${fold}, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
      python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':${fold}, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
  done
 
echo python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':-1, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':-1, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"

base_folder="'/home/fabiofabris/results_proj/go'"
feature_type="'go_features.csv'"
binary_feature="True"
  for fold in $(seq 0 9); do
      echo python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':${fold}, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
      python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':${fold}, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
  done

echo python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':-1, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"
python3 experiment.py "{'base_folder':${base_folder}, 'epochs':${epochs}, 'seed':0, 'fold':-1, 'binary_feature':${binary_feature}, 'base_model':${base_model}, 'feature_type':${feature_type}}"

epochs=1
base_model="'joined'"
base_folder="'/home/fabiofabris/results_proj_2/go+pathdip+ppi+gtex/'"
base_in_folder="'/home/fabiofabris/results_proj/'"
feature_types="['pathdip','gtex','ppi','go']"

# for fold in $(seq 0 9); do
#     echo python experiment.py "{'seed':0,'base_folder':${base_folder}, 'base_in_folder':${base_in_folder}, 'epochs':${epochs}, 'fold':${fold}, 'models_to_load':${feature_types}, 'base_model':${base_model}}"
#     python3 experiment.py "{'seed':0,'base_folder':${base_folder}, 'base_in_folder':${base_in_folder}, 'epochs':${epochs}, 'fold':${fold}, 'models_to_load':${feature_types}, 'base_model':${base_model}}"
# done

echo python experiment.py "{'seed':0,'base_folder':${base_folder}, 'base_in_folder':${base_in_folder}, 'epochs':${epochs}, 'fold':-1, 'models_to_load':${feature_types}, 'base_model':${base_model}}"
python3 experiment.py "{'seed':0,'base_folder':${base_folder}, 'base_in_folder':${base_in_folder}, 'epochs':${epochs}, 'fold':-1, 'models_to_load':${feature_types}, 'base_model':${base_model}}"

epochs=150
base_model="'full_dataset_tree'"
base_folder="'/home/fabiofabris/results/full_dataset_tree/'"
 
 for fold in $(seq 1 9); do
 	echo python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'epochs':${epochs}, 'fold':${fold}, 'base_model':${base_model}}"
 	        python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'epochs':${epochs}, 'fold':${fold}, 'base_model':${base_model}}"
 done
 
 echo python3 experiment.py "{'seed':0,'base_folder':${base_folder}, 'epochs':${epochs}, 'fold':-1, 'base_model':${base_model}}"
 python3 experiment.py "{'seed':0,'base_folder':${base_folder}, 'epochs':${epochs}, 'fold':-1, 'base_model':${base_model}}"


binary_feature="False"
base_model="'boosted_tree'"
feature_type="'base_expression_gtex_features.csv'"
base_folder="'/home/fabiofabris/results/tree_v2/gtex/'"
for fold in $(seq 0 9); do
    echo python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'fold':${fold}, 'base_model':${base_model},'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"
    python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'fold':${fold}, 'base_model':${base_model}, 'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"
done

echo python3 experiment.py "{'seed':0,'base_folder':${base_folder}, 'fold':-1, 'base_model':${base_model}, 'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"
python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'fold':-1, 'base_model':${base_model}, 'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"

binary_feature="True"
base_model="'boosted_tree'"
feature_type="'ppi_features.csv'"
base_folder="'/home/fabiofabris/results/tree_v2/ppi/'"
for fold in $(seq 0 9); do
    echo python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'fold':${fold}, 'base_model':${base_model},'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"
    python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'fold':${fold}, 'base_model':${base_model}, 'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"
done

echo python3 experiment.py "{'seed':0,'base_folder':${base_folder}, 'fold':-1, 'base_model':${base_model}, 'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"
python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'fold':-1, 'base_model':${base_model}, 'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"

binary_feature="True"
base_model="'boosted_tree'"
feature_type="'go_features.csv'"
base_folder="'/home/fabiofabris/results/tree_v2/go/'"
for fold in $(seq 0 9); do
    echo python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'fold':${fold}, 'base_model':${base_model},'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"
    python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'fold':${fold}, 'base_model':${base_model}, 'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"
done

echo python3 experiment.py "{'seed':0,'base_folder':${base_folder}, 'fold':-1, 'base_model':${base_model}, 'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"
python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'fold':-1, 'base_model':${base_model}, 'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"

binary_feature="True"
base_model="'boosted_tree'"
feature_type="'pathdipall_features.csv'"
base_folder="'/home/fabiofabris/results/tree_v2/pathdipall/'"
for fold in $(seq 0 9); do
    echo python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'fold':${fold}, 'base_model':${base_model},'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"
    python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'fold':${fold}, 'base_model':${base_model}, 'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"
done

echo python3 experiment.py "{'seed':0,'base_folder':${base_folder}, 'fold':-1, 'base_model':${base_model}, 'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"
python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'fold':-1, 'base_model':${base_model}, 'feature_type':${feature_type}, 'binary_feature':${binary_feature}}"



epochs=150
base_model="'joined_seq'"
base_folder="'/home/fabiofabris/results_bugfix/go+pathdip+ppi+gtex_seq/'"
base_in_folder="'/home/fabiofabris/results/'"
feature_types="['pathdip','gtex','ppi','go']"

for fold in $(seq 0 9); do
    echo python experiment.py "{'base_folder':${base_folder}, 'base_in_folder':${base_in_folder}, 'epochs':${epochs}, 'fold':${fold}, 'models_to_load':${feature_types}, 'base_model':${base_model}}"
    python3 experiment.py "{'base_folder':${base_folder}, 'base_in_folder':${base_in_folder}, 'epochs':${epochs}, 'fold':${fold}, 'models_to_load':${feature_types}, 'base_model':${base_model}, 'seed':0}"
done

echo python experiment.py "{'base_folder':${base_folder}, 'base_in_folder':${base_in_folder}, 'epochs':${epochs}, 'fold':-1, 'models_to_load':${feature_types}, 'base_model':${base_model}}"
python3 experiment.py "{'base_folder':${base_folder}, 'base_in_folder':${base_in_folder}, 'epochs':${epochs}, 'fold':-1, 'models_to_load':${feature_types}, 'base_model':${base_model}}"

models_to_load="['gtex', 'pathdipall', 'go', 'ppi']"
base_model="'joined_tree'"
base_folder="'/home/fabiofabris/results/tree_v2/go+pathdip+ppi+go/'"
base_in_folder="'/home/fabiofabris/results/tree_v2/'"


for fold in $(seq 0 9); do
	echo python3 experiment.py "{'seed':0, 'base_folder':${base_folder}, 'fold':${fold}, 'base_model':${base_model},'models_to_load':${models_to_load}}"
	python3 experiment.py "{'base_in_folder':${base_in_folder}, 'seed':0, 'base_folder':${base_folder}, 'fold':${fold}, 'base_model':${base_model}, 'models_to_load':${models_to_load}}"
done

