from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import header
from file_reader import *
from build_model import test_model

training_data = "data/training data/labeled_training_set.csv"
test_data = "data/test data/unlabeled_test_set.csv"

training_data_path = os.path.join(root_folder, training_data)
test_data_path = os.path.join(root_folder, test_data)

systemrun_path = os.path.join(root_folder, 'systemrun')



X, Y = read_data(training_data_path, clean=False)

# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
# train_data = [['Example sentence 1 for multilabel classification.', [1, 1, 1, 1, 0, 1]]] + [['This is another example sentence. ', [0, 1, 1, 0, 0, 0]]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle=False)
train_data = list(zip(X_train, y_train))
train_df = pd.DataFrame(train_data, columns=['text', 'labels'])
train_df = pd.DataFrame(train_data)

# eval_data = [['Example eval sentence for multilabel classification.', [1, 1, 1, 1, 0, 1]], ['Example eval senntence belonging to class 2', [0, 1, 1, 0, 0, 0]]]
eval_data = list(zip(X_test, y_test))
eval_df = pd.DataFrame(eval_data)

train_model = False
model = ''

if train_model:
	# Create a MultiLabelClassificationModel
	# model = MultiLabelClassificationModel('roberta', 'roberta-base', num_labels=6, args={'reprocess_input_data': True, 'overwrite_output_dir': True, 'num_train_epochs': 5})
	model = MultiLabelClassificationModel('albert', 'albert-base-v2', num_labels=6, args={'reprocess_input_data': True, 'overwrite_output_dir': True, 'num_train_epochs': 5})
	# You can set class weights by using the optional weight argument
	print(train_df.head())
	# Train the model
	model.train_model(train_df)
else:
	model = MultiLabelClassificationModel('albert', './outputs')

# Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(eval_df)
# print(result)
# print(model_outputs)

predictions, raw_outputs = model.predict(X_test)
test_model(y_test, predictions)

print(predictions)
print(raw_outputs)