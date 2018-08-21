# CSS490
# Yeseul(Ashley An)
# Term project to incorporate classification model with different classifiers to
# predict on the breast cancer data set from University of Wisconsin
#----------------------------------------------------------------------------------------------------------------
import pandas as pd #import pandas to read data
import numpy as np # import numpy to use linear algebra algorithms
import matplotlib.pyplot as plt # import to visualize data on the plots
from sklearn.decomposition import PCA # import to implement PCA algorithm
import seaborn as sns # import to visualize data
from sklearn.model_selection import train_test_split # use to train to split the data
from sklearn.neighbors import KNeighborsClassifier # import to implement KNN algorithms
from sklearn.model_selection import cross_val_score # import to implement cross validation score
from sklearn.metrics import classification_report, confusion_matrix # import to evaluate the algorithm
from sklearn import metrics # import to visualize metrics
from sklearn.datasets import load_breast_cancer # import to use pre-processed breast cancer data from sklearn
from sklearn.linear_model import LogisticRegression # import to implement logistic regression algorithm
from sklearn.preprocessing import StandardScaler # import to standardize and normalize data
from sklearn.tree import DecisionTreeClassifier # import to implement decision tree algorithm
from sklearn.tree import export_graphviz # import to create dot file for the decision tree
from sklearn.svm import SVC # import to use SVM algorithm
from sklearn.naive_bayes import GaussianNB # import to implement gaussianNB for the naive bayes algorithm

# import the data file and preprocessing the breast cancer data set
# drops the unused columns such as Unnamed or id columns
cancer = pd.read_csv('data.csv')
cancer.drop('Unnamed: 32', axis=1, inplace=True)
cancer.drop('id', axis=1, inplace=True)

# Map the labels to the binary number. Malignant to 1 and Benign to 0
cancer['diagnosis'] = cancer['diagnosis'].map({'M':1, 'B':0})

X = cancer.iloc[:, 1:] #Dataframe
y = cancer.iloc[:, 0]  # Series
diag = cancer.iloc[:, 0:1]

# Plot the correlation heat map of 30 features
corr = cancer.iloc[:, 1:].corr()
colormap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure(figsize= (14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f', annot_kws={'size': 8},
             cmap = colormap, linewidths=0.1, linecolor='white')
plt.title('Correlation of 30 Features', y=1.05, size=15)
plt.savefig('FeatureCorr.png')

# Standardize and fit data. Also, drops the columns that have high collinearity
# Drop the unnecessary columns based on the correlation map
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled)
X_dropped = X_scaled.drop(X_scaled.columns[[2, 3, 12, 13, 22, 23]], axis=1)

# Standardize data with PCA. n_compoments=0.95 means deciding on the number of principal components which
# covers 95% of the data set
# There are 11 principal components created through PCA
pca = PCA(n_components=0.95)
p_pca = pca.fit_transform(X_dropped)

#print(p_pca.shape)
#print(pca.explained_variance_ratio_)
#print(pca.explained_variance_ratio_.sum())

names = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'diagnosis']
Xy = pd.DataFrame(np.hstack([p_pca, np.asmatrix(diag)]),columns=names)
sns.lmplot("PC1", "PC2", hue="diagnosis", data=Xy, fit_reg=False, markers=["o", "x"],palette="Set1")
plt.savefig('PCA.png')

# re-instantiate X based on the PCA analysis result
X = np.asmatrix(Xy.iloc[:, 0:11])
y = cancer.iloc[:, 0].values

# splits the data set into 80% and 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# KNN Algorithm -------------------------------------------------------------------------------------------
knn = KNeighborsClassifier()

# Between range 1 - 40, we find the optimal K that has highest sensitivity
krange = list(range(1,40))
kscores =[]

for i in krange:
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='recall')
    kscores.append(scores.mean())

#print(np.round(kscores, 3))

# Visualize sensitivity-K values plot on the graph to find the optimal K
my_dpi = 96
plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi=my_dpi)
plt.plot(krange, kscores, color='blue')
plt.xlabel('K-values')
plt.ylabel('CVS Mean')
plt.title('CVS and K-Values')
plt.savefig('KNN_Recall.png')

# Based on the visualization, it found that optimal K can be 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_y_predict = knn.predict(X_test)

print('\n Classification report: KNN \n')
print(classification_report(y_test, knn_y_predict))
print('\n Confusion Matrix: KNN \n')
print(confusion_matrix(y_test, knn_y_predict))

matrix = metrics.confusion_matrix(y_test, knn_y_predict)
f,ax = plt.subplots(figsize=(5, 4))
sns.heatmap(matrix, annot=True, linewidths=.5, fmt= '.1f',ax=ax);
plt.title('KNN Confusion Matrix')
plt.savefig('KNN_matrix.png')

print('Accuracy of KNN n-5 on the training set: {:.3f}' .format(knn.score(X_train, y_train)))
print('Accuracy of KNN n-5 on the test set: {:.3f}' .format(knn.score(X_test, y_test)))
print()
print ('-------------------------------------------------------------------------------')

# Logistic Regression -----------------------------------------------------------------------------------------------
# instantiate logistic regression classifier
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# The default C parameter of logistic regression is C=1
print('Accuracy of Logistic Regression(C=1) on the training subset: {:.3f}' .format(log_reg.score(X_train, y_train)))
print('Accuracy of Logistic Regression(C=1) on the test subset: {:.3f}' .format(log_reg.score(X_test, y_test)))
print()

# Result to see when the C parameter changes to 100
log_reg100 = LogisticRegression(C=100)
log_reg100.fit(X_train, y_train)
print('Accuracy of Logistic Regression(C=100) on the training subset: {:.3f}' .format(log_reg100.score(X_train, y_train)))
print('Accuracy of Logistic Regression(C=100) on the test subset: {:.3f}' .format(log_reg100.score(X_test, y_test)))
print()

# Result to see when the C parameter changes to 0.01
log_reg001 = LogisticRegression(C=0.01)
log_reg001.fit(X_train, y_train)
print('Accuracy of Logistic Regression(C=0.01) on the training subset: {:.3f}' .format(log_reg001.score(X_train, y_train)))
print('Accuracy of Logistic Regression(C=0.01) on the test subset: {:.3f}' .format(log_reg001.score(X_test, y_test)))
print()

# Choose C=1 because it has the most optimal result
log_reg = LogisticRegression(C=1)
log_reg.fit(X_train, y_train)

log_y_predict = log_reg.predict(X_test)
print('\n Classification report: Logistic Regression(C=1) \n')
print(classification_report(y_test, log_y_predict))
print('\n Confusion matrix: Logistic Regression(C=1) \n')
print(confusion_matrix(y_test, log_y_predict))

matrix = metrics.confusion_matrix(y_test, log_y_predict)
f,ax = plt.subplots(figsize=(5, 4))
sns.heatmap(matrix, annot=True, linewidths=.5, fmt= '.1f',ax=ax);
plt.title('Logistic Regression Confusion Matrix')
plt.savefig('Logistic_matrix.png')

# The default C parameter of logistic regression is C=1
print('Accuracy of Logistic Regression(C=1) on the training subset: {:.3f}' .format(log_reg.score(X_train, y_train)))
print('Accuracy of Logistic Regression(C=1) on the test subset: {:.3f}' .format(log_reg.score(X_test, y_test)))
print()
print ('-------------------------------------------------------------------------------')

# Decision Tree -----------------------------------------------------------------------------------------------------
dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train, y_train)

# Shows the classifier is overfitting because training subset has 1.000 score
print('Accuracy of Decision Tree on the training subset: {:.3f}' .format(dtree.score(X_train, y_train)))
print('Accuracy on Decision Tree on the testing subset: {:.3f}' .format(dtree.score(X_test, y_test)))

dtree = DecisionTreeClassifier(max_depth=4, random_state=0)
dtree.fit(X_train, y_train)

dtree_y_predict = dtree.predict(X_test)
print('\n Classification report: Decision Tree \n')
print(classification_report(y_test, dtree_y_predict))
print('\n Confusion matrix: Decision Tree \n')
print(confusion_matrix(y_test, dtree_y_predict))

matrix = metrics.confusion_matrix(y_test, dtree_y_predict)
f,ax = plt.subplots(figsize=(5, 4))
sns.heatmap(matrix, annot=True, linewidths=.5, fmt= '.1f',ax=ax);
plt.title('Decision Tree Confusion Matrix')
plt.savefig('DecisionTree_matrix.png')

print('Accuracy of Decision Tree(Depth=4) on the training subset: {:.3f}' .format(dtree.score(X_train, y_train)))
print('Accuracy on Decision Tree(Depth=4) on the testing subset: {:.3f}' .format(dtree.score(X_test, y_test)))
print()
print ('-------------------------------------------------------------------------------')

# export dot file that is used to draw decision tree for decision tree
# the decision tree shows how the predictions are made based on the data set
# Since the depth is 4, it shows depth of 4 decision tree
# Also shows the importance/weight each feature carries

d_names = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11']

c = load_breast_cancer()
export_graphviz(dtree, out_file='cancertree.dot', class_names=['malignant', 'benign'], feature_names=d_names, impurity=False, filled=True)

n_features = X.data.shape[1]
my_dpi = 96
plt.figure(figsize = (1500/my_dpi, 800/my_dpi), dpi=my_dpi)
plt.barh(range(n_features), dtree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), d_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Importance of Features')
plt.savefig('feature_importance.png')

# Naive Bayes -----------------------------------------------------------------------------------------
naive = GaussianNB()
naive.fit(X_train, y_train)

naive_predict = naive.predict(X_test)

print('\n Classification report: Naive Bayes \n')
print(classification_report(y_test, naive_predict))
print('\n Confusion matrix: Naive Bayes \n')
print(confusion_matrix(y_test, naive_predict))

matrix = metrics.confusion_matrix(y_test, naive_predict)

f,ax = plt.subplots(figsize=(5, 4))
sns.heatmap(matrix, annot=True, linewidths=.5, fmt= '.1f',ax=ax);
plt.title('Naive Bayes Confusion Matrix')
plt.savefig('NavieBayes_matrix.png')

print('Accuracy of Naive Bayes on the training subset: {:.3f}' .format(naive.score(X_train, y_train)))
print('Accuracy of Naive Bayes on the testing subset: {:.3f}' .format(naive.score(X_test, y_test)))
print()
print ('-------------------------------------------------------------------------------')

# SVM --------------------------------------------------------------------------------------------------
svm = SVC(probability=True)
svm.fit(X_train, y_train)

svm_predict = svm.predict(X_test)

print('\n Classification report: SVM \n')
print(classification_report(y_test, svm_predict))
print('\n Confusion matrix: SVM \n')
print(confusion_matrix(y_test, svm_predict))

matrix = metrics.confusion_matrix(y_test, svm_predict)

#plt.figure(figsize=(14,14))
f,ax = plt.subplots(figsize=(5, 4))
sns.heatmap(matrix, annot=True, linewidths=.5, fmt= '.1f',ax=ax);
plt.title('SVM Confusion Matrix')
plt.savefig('SVM_matrix.png')

print('Accuracy of SVM on the training subset: {:.3f}' .format(svm.score(X_train, y_train)))
print('Accuracy of SVM on the testing subset: {:.3f}' .format(svm.score(X_test, y_test)))
print()
print ('-------------------------------------------------------------------------------')

# Plots ROC curves for all classifiers -------------------------------------------------------------------------
y_pred_knn = knn.predict_proba(X_test)[:,1]
y_pred_log = log_reg.predict_proba(X_test)[:,1]
y_pred_dtree = dtree.predict_proba(X_test)[:,1]
y_pred_naive = naive.predict_proba(X_test)[:,1]
y_pred_svm = svm.predict_proba(X_test)[:,1]

models=[y_pred_knn, y_pred_log, y_pred_dtree, y_pred_naive, y_pred_svm]
label=['KNN', 'Logistic', 'D_Tree', 'Naive Bayes', 'SVM']

plt.figure(figsize=(10,8))
range = np.arange(5)

for i in range:
    f_pr, t_pr, thresholds = metrics.roc_curve(y_test, models[i])
    print('Classifier:', label[i])
    print('Thresholds:', np.round(thresholds, 3))
    print('Tpr       :', np.round(t_pr, 3))
    print('Fpr       :', np.round(f_pr,3))
    plt.plot(f_pr, t_pr, label=label[i])

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Courve for Each Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc=4,)
plt.savefig('ROC.png')