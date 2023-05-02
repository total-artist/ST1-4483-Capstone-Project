#HEre we are importing varius modules.
import sklearn
from sklearn.utils import shuffle
from sklearn import datasets
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Load libraries
import numpy
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

#Here we import the titanic data file using hte pd.read_csv
titanic_data = pd.read_csv("train.csv")

#Here we set the attribute to be predicted.
predict = "Survived"

# Dataset/Column to be Predicted, X is all attributes and y is the features
#x = np.array(titanic_data.drop([predict], 1)) # Will return a new data frame that doesnt have hd in it
#y = np.array(titanic_data[predict])
#"le" is a variable used to create an instance of hte "LabelEncoder" class which is used to encode category variables into numeric values.
le = preprocessing.LabelEncoder()

#The "passengerID" variable is used to apply label encoding on the "PassengerId" column.
passengerId = le.fit_transform(list(titanic_data["PassengerId"]))

#The "pclass" variable is used to apply label encoding on the Pclass column
pclass = le.fit_transform(list(titanic_data["Pclass"]))

#The "name" variable is used to apply label encoding on the Name columne
name = le.fit_transform(list(titanic_data["Name"]))

#The "sex" variable is used to apply label encoding on the Sex column
sex = le.fit_transform(list(titanic_data["Sex"]))

#The "age" variable is used to apply label encoding on the Age column
age = le.fit_transform(list(titanic_data["Age"]))

#The "sibSp" variable is used to apply label encoding on the SibSp column
sibSp = le.fit_transform(list(titanic_data["SibSp"]))

#The "parch" variable is used to apply label encoding on the Parch column
parch = le.fit_transform(list(titanic_data["Parch"]))

#The "ticket" variable is used to apply label encoding on the Ticket column
ticket = le.fit_transform(list(titanic_data["Ticket"]))

#The "fare" variable is used to apply label encoding on the Fare column
fare = le.fit_transform(list(titanic_data["Fare"]))

#The "cabin" variable is used to apply label encoding on the Cabin column
cabin = le.fit_transform(list(titanic_data["Cabin"]))

#The "embarked" variable is used to apply label encoding on the Embarked column
embarked = le.fit_transform(list(titanic_data["Embarked"]))

#The "survived" variable is used to apply label encoding on the Survived column
survived = le.fit_transform(list(titanic_data["Survived"]))

#This x variable is used for zipping the encoded values together using hte zip() function. These will be used as inputs.
x = list(zip(passengerId, pclass, name, sex, age, sibSp, parch, ticket, fare, cabin, embarked))

#This y variable is used to create a list of encoded values to be used for the prediction.
#this is the output.
y = list(survived)


# num of folds means that the data set will be split into 4 equal parts for training/testing.
num_folds = 4

# the seed is used to for random number generation.
seed = 7

#we use the term accuracy to define how accurate our model is
scoring = 'accuracy'

# Model Test/Train
#We import the sklearn.model_selection to split hte data into training and test sets for model evaluation.
import sklearn.model_selection

#x contains the attributes for that will be used in predictions and y contains the target variables.
#The test_size parameter is used to specify the percentage of the data that will be used for testing which is 20%
#The random_state parameter is used to set the random seed to 7 as declared above for random number generation.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
#this splits 20% of our data into test samples.

# Check with  different Scikit-learn classification algorithms
#Here we create an empty list to store the models.
models = []

#These lines of code add/append the models to the empty list.
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn

#HEre we create a list to store accuracy scores for each model.
results = []

#HEre we create a list store the name of the models.
names = []

#The for loop will be used to iterate through the list of models and unpacks each tuple into name and model.
for name, model in models:
	# This variable is used to create an instance of the Kfold class with the number of folds and the number of seeds as parameters
	kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
	cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	msg += '\n'
	print(msg)

# Compare Algorithms' Performance
#Here we will use a box plot diagram to do that.
fig = pyplot.figure()
fig.suptitle('Algorithm/model Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# Make predictions on validation/test dataset
#the code below initializes the models.
dt = DecisionTreeClassifier()
nb = GaussianNB()
gb = GradientBoostingClassifier()
rf = RandomForestClassifier()

#this assigns the GradiaentBoostingClassifier model to the variable best_model
#We see this model performed the best so we assigned it to a variable for use later.
best_model = gb

#Here we fits the training data to the best model, this trains the model on the training data.
best_model.fit(x_train, y_train)

#Here we make predictions on the tested data using hte GradientBoostingClassifier model.
y_pred = best_model.predict(x_test)
model_accuracy = accuracy_score(y_test, y_pred)
print("Best Model Accuracy Score on Test Set:", model_accuracy)

#Model Evaluation Metric 1
#this code will be used to generate a report on the performance of hte classification model
print(classification_report(y_test, y_pred))

#Model Evaluation Metric 2
#Confusion matrix
#Here we import the confusiong_matrix and ConfusionMatrixDisplay classes which will be used for performance evaluation.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#The cm variable will be used to calculate teh confusion matrix for the predicted labels (y_pred) and true labels (y_test)
cm = confusion_matrix(y_test, y_pred)

#The disp variable will be used to create an instance of the ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

#Here we plot it and show it.
disp.plot()
plt.show()

#Model Evaluation Metric 3

#these 2 lines of code import the roc_auc_score and roc_curve functions necessary to calculate ROC-AUC score the ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#here we assign the best model "GradientBoostingClassifier" to a variable
best_model = gb

#Here we are fitting training data into the best_model variable
best_model.fit(x_train, y_train)

#This line is used to calculate teh ROC-AUC score for the best_model
rf_roc_auc = roc_auc_score(y_test,best_model.predict(x_test))

#Here we calculate the false positive rate, true positive rate and thresholds for the ROC-Curve
fpr,tpr,thresholds = roc_curve(y_test, best_model.predict_proba(x_test)[:,1])

#HEre we create a new figure for the ROC curve plot.
plt.figure()

#coordinates and labels are added below.
plt.plot(fpr,tpr,label = 'Random Forest(area = %0.2f)'% rf_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('LOC_ROC')
plt.show()

#Check actual/ground truth vs predicted diagnosis
#The purpose of this loop is print out a report on predicted values,
#actual values, and the corresponding data points for each iteration
for x in range(len(y_pred)):
	print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x],)
