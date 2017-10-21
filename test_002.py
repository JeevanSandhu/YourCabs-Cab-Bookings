import pandas
from sklearn import svm
from sklearn.metrics import accuracy_score
from time import time

clf = svm.SVC()

#######################
#### TRAINING
######################

train_data = pandas.read_csv("data/Kaggle_YourCabs_training.csv")
print('Train Data read!')

true_labels = train_data['Car_Cancellation']

col_names = list(train_data)
col_names.remove('Car_Cancellation')
col_names.remove('Cost_of_error')
col_names.remove('from_date')
col_names.remove('to_date')
col_names.remove('booking_created')

features = train_data[col_names].astype(float)
features = features.fillna(0)

t0 = time()
clf.fit(features, true_labels)
tt = time()-t0
print("Classifier Fit in {} seconds".format(round(tt,3)))

t0 = time()
predicted_labels = clf.predict(features)
tt = time() - t0
print("Assigned labels for training_data in {} seconds".format(round(tt,3)))

accuracy_score = accuracy_score(true_labels, predicted_labels)
print("\n\nAccuracy {} %".format(round(accuracy_score*100,3)))

#######################
#### TESTING
######################

test_data = pandas.read_csv('data/Kaggle_YourCabs_score.csv')
print('Test Data read!')

col_names = list(test_data)
col_names.remove('from_date')
col_names.remove('to_date')
col_names.remove('booking_created')
col_names.remove('Unnamed: 18')
col_names.remove('Unnamed: 19')

features = test_data[col_names].astype(float)
features = features.fillna(0)

t0 = time()
predicted_labels = clf.predict(features)
tt = time() - t0
print("Assigned labels for training_data in {} seconds".format(round(tt,3)))


#######################
#### WRITE RESULTS TO FILE
######################

with open('test_002-submission.csv', 'w') as f:
	f.write('id,Car_Cancellation\n')

with open('test_002-submission.csv', 'a') as f:
	for i in range(0,len(test_data)):
		f.write('{},{}\n'.format(test_data['id'][i], predicted_labels[i]))