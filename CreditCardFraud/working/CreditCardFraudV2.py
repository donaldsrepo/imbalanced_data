#!/usr/bin/env python
# coding: utf-8

# <h1 align='center' style='color:purple'>Credit Card Fraud - Imbalanced Data Set</h1>

# **Use Case:** Credit Card Fraud Detection
# 
#     Compare different common algorithms, develop and optimize a new 2 sequential/consecutive model algorithm to see if this can give better results
#     
# **Author:** Donald Stierman - Senior Data Scientist
# 
# **Details:** Imbalanced data can cause issues with most Machine Learning Algorithms and Neural Networks. To alleviate this, I choose to down-sample the training data to use as the input dataset. After creating the down-sampled dataset, I ran this through several different common model algorithms, including a new modeling technique I developed specifically for imbalanced data. I got this idea after reading about some highly effective Healthcare screening solutions currently in use. I.E. Breast Cancer detection in women (see comments: below). If a mammogram comes back positive, we already know that there will be a lot of false positives (benign tumors, scars, etc). Usually the doctor will follow up with a 2nd test, such as biopsy. This will screen out the false positives leaving mostly true positives (cancerous tissue). This same idea can possibly be applied to credit care fraud. We want to catch all true cases of fraud (fraud prevention), to be compliant with government regulations, and additionally not create a huge workload of false cases to be investigated (cost control).
# 
# comments:
# 
# **Here are some different ways to explain the methodology used in the Healthcare use case:**
# 
# *1st test (high specificity) -> 2nd test (high sensitivity) -> Only treat cancerous tissue
# 
# *TN/(TN + FP) is high ~ 1    -> TP/(TP + FN) is high ~ 1    -> Find all Positive cases
# 
# *catch all possible cases/remove healthy patients -> remove all false flags -> high confidence in Positive result/few missed positives
# 
# 
# This same methodology can be applied to Credit Card Fraud detection
# 
# Link to code repo at Github:
# 
# https://github.com/donaldstierman/imbalanced_data
# 
# **Models used:**<pre>
#     Logistic Regression
#     Random Forest
#     Gradient Boosted Decision Trees
#     Customized 2 Step Gradient Boosted Decision Trees
#     Deep Neural Network
#     1D Convolutional Neural Network
#     AutoEncoder
# </pre>
# **Goal:** 
# For this example, I chose 2 metrics to optimize, ROC/AUC and best "macro avg recall". I chose these because in the health care example, it is better to catch all cancer patients, even if it means more tests are performed. To compare the results, first objective is to find the best overall model (lowest mislabelled predictions), second is to find the model that has a low number of false negatives (faudulent transactions that are missed) without having too many false positives (genuine transactions that are needlessly investigated)
# <pre>
#     1) Compare the AUC to find the most robust model of the single step models. However, the value of this metric cannot be calculated directly on the 2 step model, so we need to use #2 below for final comparison
#     2) Maximize the Sensitivity (higher priority) or reduce the number of False Negatives (FN/TP ratio) and maximize the Specificity (lower priority) to control the number of tests performed in the 2nd step. I.E. catch all the fraudulent transaction even if there are false flags (false positives).
# </pre>
# **Results:** The Customized 2 Step model has the best results overall, by only a slight margin. 
#                           
#                                 AUC    Specificity/Sensitivity
#                           
#     Logistic Regression         .967    .95/.87
#     Random Forest               .977    .97/.89  **best AUC**
#     Gradient Boosted Tree       .976    .99/.84
#     Customized 2 Step GB Trees  NA      .99/.93  **best overall**
#     Deep Neural Network         .973    .95/.92  **2nd best overall**
#     AutoEncoder                 .954    .88/.93    
#     
#     
# Final Results: ROC Curve comparision
# 
# <!-- to save file to html for uploads, use this command: jupyter nbconvert --to html --template basic CreditCardFraud.ipynb -->

# In[1]:


# Import Libraries
# try some of these ideas: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
import numpy as np
import pandas as pd

import os                                                                                                            
import matplotlib as mpl                                                                                             
if os.environ.get('DISPLAY','') == '':                                                                               
    print('no display found. Using non-interactive Agg backend')                                                     
    mpl.use('Agg')                                                                    
        
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas_profiling as pp
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
import zipfile

import tensorflow as tf


# Always like to include a timer function to see where my code is running slow or taking most of the run time

# In[2]:


class MyTimer():
    # usage:
    #with MyTimer():                            
    #    rf.fit(X_train, y_train)
    
    def __init__(self):
        self.start = time.time()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        runtime = end - self.start
        msg = 'The function took {time} seconds to complete'
        print(msg.format(time=runtime))


# In[3]:


def CalcPct(df,title):
    unique_elements, counts_elements = np.unique(df, return_counts=True)
    calc_pct = round(counts_elements[1]/(counts_elements[0]+counts_elements[1]) * 100,6)
    print(title)
    print(np.asarray((unique_elements, counts_elements)))
    return calc_pct


# In[4]:


colab = os.environ.get('COLAB_GPU', '10')
if (int(colab) == 0):
    from google.colab import drive
    drive.mount('/content/drive')  
else:
    print("")


# Setup to run on Google Colab and Kaggle platforms

# In[5]:


# Check if Google Colab path exists
if os.path.exists("/content/drive/My Drive/MyDSNotebooks/Imbalanced_data/input/creditcardzip") :
    # Change the current working Directory    
    os.chdir("/content/drive/My Drive/MyDSNotebooks/Imbalanced_data/input/creditcardzip")
# else check if Kaggle/local path exists
elif os.path.exists("../input/creditcardzip") :
    # Change the current working Directory    
    os.chdir("../input/creditcardzip")
else:
    print("Can't change the Current Working Directory") 
print("Current Working Directory " , os.getcwd())


# In[6]:


verbose=0
# Load the Data Set
df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
#off line data source for backup
#df = pd.read_csv('creditcard.csv')


# Public Credit Card Dataset. This is financial data, and is considered to be sensitive so it is "encrypted" through the use of PCA to protect privacy. Only the Time and Dollar columns are intact after the "encryption"
# 
# Doing some initial data exploration

# In[7]:


# Check the data, make sure it loaded okay
print(df.head())


# In[8]:


# Check the datatypes of the Data set 
df.info()


# In[9]:


# Check the Uniqueness
df.nunique()


# In[10]:


# Check for missing data
df.isnull().sum()


# In[11]:


# Check basic Statistics

df.describe(include ='all')


# In[12]:


# Check the Class Imbalance of the Data 

df['Class'].value_counts()


# In[13]:


# Histograms of the features
# most of the data has a quasi-normal/gaussian distribution

df.hist(bins=20, figsize=(20,15))
plt.show()


# Look at cross correlations between features. Most models will be fine with collinearity, but good to know this in any case. Most of my input is numerical, and my label is binary classification, so I can choose the Anova or Kendall's method. I will try the Kendall tau-b method first. This method will sort the 2 columns and compare if the X is always > or < Y. If so, the tau-b value will be 1.
# 
# Some key points to remember:
# Kendall’s Tau: Calculations based on concordant and discordant pairs. Insensitive to error. P values are more accurate with smaller sample sizes. Good resource can be found here: https://online.stat.psu.edu/stat509/node/158/
# 
# This image shows which method you should choose based on your dataset:
# 
# ![image.png](attachment:2deb1518-2274-4853-9694-97f893bfa5b0.png)

# In[14]:


f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(method='kendall'), fignum=f.number) # pearson or spearman are also available
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title("Kendall's Correlation Matrix Full Data Set", fontsize=16)
plt.show()


# V21 and V22 show the highest tau-b score, will investigate this relationship later

# In[15]:


#try some data cleansing, Amount has a few high values, so try using the log of that column instead.

temp_df = df.copy()
temp_df = temp_df.drop(['Time'], axis=1)
temp_df['Log_Amount'] = np.log(temp_df.pop('Amount')+0.001)
df = temp_df.copy()


# Divide the dataset into features and labels and then into Train, Test and Validate datasets

# In[16]:


# divide full data into features and label
spl1 = 0.3
spl2 = 0.3
X = df.loc[:, df.columns != 'Class']
y = df.loc[:, df.columns == 'Class']
OrigPct = CalcPct(y,"Original")

strat = True
if (strat == True):
    stratify=y['Class']
else:
    stratify="None"
# create train, test and validate datasets

# first split original into Train and Test+Val
X_train, X_test1, y_train, y_test1 = train_test_split(X,y, test_size = spl1, random_state = None, shuffle=True, stratify=stratify)
# then split Test+Val into Test and Validate
# Validate will only be used in the 2 Model system (explained below)
X_test, X_val, y_test, y_val = train_test_split(X_test1,y_test1, test_size = spl2, random_state = None, shuffle=True)


# In[17]:


f = plt.figure(figsize=(16, 12))
plt.matshow(X_train.corr(method='kendall'), fignum=f.number) # pearson or spearman are also available
plt.xticks(range(X_train.shape[1]), X_train.columns, fontsize=14, rotation=45)
plt.yticks(range(X_train.shape[1]), X_train.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title("Kendall's Correlation Matrix Initial Train Set", fontsize=16)
plt.show()


# In[18]:


# prepare data for model, need to do this normalization and clipping separately for X_train, X_test and X_val 
# to avoid any contamination between Train and Test/Validate datasets. Also need to keep the order of the rows to match
# the y label dataframes

sc = StandardScaler()

scaled_features = StandardScaler().fit_transform(X_train.values)
X_train = pd.DataFrame(scaled_features, index=X_train.index, columns=X_train.columns)
scaled_features = StandardScaler().fit_transform(X_test.values)
X_test = pd.DataFrame(scaled_features, index=X_test.index, columns=X_test.columns)
scaled_features = StandardScaler().fit_transform(X_val.values)
X_val = pd.DataFrame(scaled_features, index=X_val.index, columns=X_val.columns)

# handle any extreme fliers, set to 5 or -5
X_train = np.clip(X_train, -5, 5)
X_test = np.clip(X_test, -5, 5)
X_val = np.clip(X_val, -5, 5)


# In[19]:


f = plt.figure(figsize=(16, 12))
plt.matshow(X_train.corr(method='kendall'), fignum=f.number) # pearson or spearman are also available
plt.xticks(range(X_train.shape[1]), X_train.columns, fontsize=14, rotation=45)
plt.yticks(range(X_train.shape[1]), X_train.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title("Kendall's Correlation Matrix Normalized [and Clipped] Train Set", fontsize=16)
plt.show()


# In[20]:


# Check basic Statistics after normalizing and clipping data

X_train.describe(include ='all')


# In[21]:


class_names=[0,1] # name  of classes 1=fraudulent transaction

y_val['Class'].value_counts()

TrainPct = CalcPct(y_train,"Train")
TestPct = CalcPct(y_test,"Train")
ValPct = CalcPct(y_val,"Train")
zeros, ones = np.bincount(y_train['Class'])


# Investigate the high tau-b value between V21 and V22

# In[22]:


# Form np arrays of labels and features for jointplot charts

train_labels = np.array(y_train).flatten()
bool_train_labels = train_labels != 0 # has an extra ,1 in the bool_train_labels.shape
val_labels = np.array(y_val)
test_labels = np.array(y_test)
train_features = np.array(X_train)
val_features = np.array(X_val)
test_features = np.array(X_test)

pos_df = pd.DataFrame(train_features[ bool_train_labels], columns = X.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns = X.columns)
sns.jointplot(pos_df['V21'], pos_df['V22'],
              kind='hex', xlim = (-5,5), ylim = (-5,5))
plt.suptitle("Positive distribution")
sns.jointplot(neg_df['V21'], neg_df['V22'],
              kind='hex', xlim = (-5,5), ylim = (-5,5))
_ = plt.suptitle("Negative distribution")


# V21 shows a slight one-sided tail, however Kendall's correlation test is good to use here as it is a non-parametric test and can handle non-gaussian distributions like this

# For a imbalanced sampling strategy, I will be using undersampling in my project as i think this is the best approach for this type of data

# In[23]:


# find the number of minority (value=1) samples in our train set so we can down-sample our majority to it
yes = len(y_train[y_train['Class'] ==1])

# retrieve the indices of the minority and majority samples 
yes_ind = y_train[y_train['Class'] == 1].index
no_ind = y_train[y_train['Class'] == 0].index

# random sample the majority indices based on the amount of 
# minority samples
new_no_ind = np.random.choice(no_ind, yes, replace = False)

# merge the two indices together
undersample_ind = np.concatenate([new_no_ind, yes_ind])

# get undersampled dataframe from the merged indices of the train dataset
X_train = X_train.loc[undersample_ind]
y_train = y_train.loc[undersample_ind]

y_train = np.array(y_train).flatten()


# In[24]:


f = plt.figure(figsize=(16, 12))
plt.matshow(X_train.corr(method='kendall'), fignum=f.number) # pearson or spearman are also available
plt.xticks(range(X_train.shape[1]), X_train.columns, fontsize=14, rotation=45)
plt.yticks(range(X_train.shape[1]), X_train.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title("Kendall's Correlation Matrix Undersampled, Normalized [and Clipped] Train Set", fontsize=16)
plt.show()


# Create some calculation and visualization functions to show the results

# In[25]:


def visualize(Actual, Pred, Algo):
    #Confusion Matrix
    cnf_matrix=metrics.confusion_matrix(Actual, Pred) #

    #Visualize confusion matrix using heat map

    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix: '+Algo, y=1.1) 
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


# In[26]:


def display_metrics(model_name, train_features, test_features, train_label, test_label, pred, algo):
    model_probs = model_name.predict_proba(test_features)
    n = model_name.predict_proba(test_features).shape[1]-1
    model_probs = model_probs[:, n]
    try:
        print(model_name.score(test_features, test_label)) 
        print("Accuracy score (training): {0:.3f}".format(model_name.score(train_features, train_label))) 
        print("Accuracy score (validation): {0:.3f}".format(model_name.score(test_features, test_label))) 
    except Exception as e:
        print("error")  
    try:
        print(pd.Series(model_name.feature_importances_, index=train_features.columns[:]).nlargest(10).plot(kind='barh')) 
    except Exception as e:
        print("error") 
    print("Confusion Matrix:")
    tn, fp, fn, tp = confusion_matrix(test_label, pred).ravel()
    total = tn+ fp+ fn+ tp 
    print("false positive pct:",(fp/total)*100) 
    print("tn", " fp", " fn", " tp") 
    print(tn, fp, fn, tp) 
    print(confusion_matrix(test_label, pred)) 
    print("Classification Report") 
    print(classification_report(test_label, pred))
    print("Specificity =", tn/(tn+fp))
    print("Sensitivity =", tp/(tp+fn))
    y=np.reshape(test_label.to_numpy(), -1)
    fpr, tpr, thresholds = metrics.roc_curve(y, model_probs, pos_label=1)
    cm_results.append([algo, tn, fp, fn, tp])
    cr_results.append([algo, classification_report(test_label, pred)])
    roc.append([algo, fpr, tpr, thresholds])
    # AUC score should be (Sensitivity+Specificity)/2
    print(algo + ':TEST | AUC Score: ' + str( round(metrics.auc(fpr, tpr),3 )))
    return tn, fp, fn, tp


# In[27]:


def auc_roc_metrics(model, test_features, test_labels, algo): # model object, features, actual labels, name of algorithm
    # useful for imbalanced data
    ns_probs = [0 for _ in range(len(test_labels))]
    # predict probabilities
    model_probs = model.predict_proba(test_features)
    # keep probabilities for the positive outcome only
    n = model.predict_proba(test_features).shape[1]-1
    model_probs = model_probs[:, n]  
    model_auc = auc_roc_metrics_plots(model_probs, ns_probs, test_labels, algo) 
    return model_auc


# In[28]:


def auc_roc_metrics_plots(model_probs, ns_probs, test_labels, algo):
    
    # calculate scores
    ns_auc = roc_auc_score(test_labels, ns_probs) # no skill
    model_auc = round(roc_auc_score(test_labels, model_probs), 3)

    # summarize scores
    print('%10s : ROC AUC=%.3f' % ('No Skill',ns_auc))
    print('%10s : ROC AUC=%.3f' % (algo,model_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(test_labels, ns_probs)
    # NameError: name 'ns_probs' is not defined
    model_fpr, model_tpr, _ = roc_curve(test_labels, model_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(model_fpr, model_tpr, marker='.', label='%s (area = %0.2f)' % (algo, model_auc))
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    pyplot.title('Receiver Operating Characteristic curve')
    # show the plot
    pyplot.show()
    return model_auc


# In[29]:


# Define our custom loss function
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


# In[30]:


def prediction_cutoff(model, test_features, cutoff):
    model.predict_proba(test_features)
    # to get the probability in each class, 
    # for example, first column is probability of y=0 and second column is probability of y=1.

    # the probability of being y=1
    prob1=model.predict_proba(test_features)[:,1]
    predicted=[1 if i > cutoff else 0 for i in prob1]
    return predicted


# In[31]:


metrics_results = {}
roc = []
cm_results = []
cr_results = []


# In[32]:


X_train.hist(bins=20, figsize=(20,15))
plt.show()


# run Logistic Regression model first

# In[33]:


lr = LogisticRegression()
#lr = LogisticRegression(solver='lbfgs')

lr.fit(X_train, y_train)
#lr_Pred = lr.predict(X_test)
# or
lr_Pred = prediction_cutoff(lr, X_test, 0.5) # 0.5 is the default cutoff for a logistic regression test


# Show the results of this model

# In[34]:


print(metrics.accuracy_score(y_test, lr_Pred))
tn, fp, fn, tp = display_metrics(lr, X_train, X_test, y_train, y_test, lr_Pred, 'LR')
visualize(y_test, lr_Pred, 'LR') # actual labels vs predicted labels
lr_auc = auc_roc_metrics(lr, X_test, y_test, 'LR')
metrics_results['lr'] = lr_auc


# In[35]:


# useful for unbalanced data, maybe include later in metrics summary for all models

lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_Pred)
lr_f1, lr_auc = f1_score(y_test, lr_Pred), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# Next try the Random Forest model

# In[36]:


#rf = RandomForestClassifier(n_estimators = 1000)

# from my other blog on optimizing models using cross validation
rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=None, max_features=None,
                       max_leaf_nodes=None, max_samples=0.8,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=10, min_samples_split=20,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

rf.fit(X_train, y_train, sample_weight=np.where(y_train == 1,1.0,1.0).flatten())

rf_Pred=rf.predict(X_test)


# Show the results of this model

# In[37]:


#print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, rf_Pred))
tn, fp, fn, tp = display_metrics(rf, X_train, X_test, y_train, y_test, rf_Pred, 'RF')
visualize(y_test, rf_Pred, 'RF')
rf_auc = auc_roc_metrics(rf, X_test, y_test, 'RF')
metrics_results['rf'] = rf_auc


# There is some variability in the results from run to run, due to random sampling and imbalanced data. This time, LogisticRegression has better prediction capability, the RandomForestClassifier test has a lot more mistakes in the False Positive category, and even a few more mistakes in the False Negative category.

# Now lets try a GradientBoosting Algorithm

# In[38]:


#setup model parameters, change some of the defaults based on benchmarking
gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=5, 
                                    max_depth=3, random_state=None, subsample = 0.5, criterion='mse', 
                                    min_samples_split = 10, min_samples_leaf = 10)

#default fit model
#gb_clf.fit(X_train, y_train)

#since a false negative is much more likely than a false positive, we should weight them accordingly
gb_clf.fit( X_train, y_train, sample_weight=np.where(y_train == 1,1.0,1.0) ) #  fn = 12 and fp = 1057
# no weights gives worse false positive counts
#gb_clf.fit( X_train, y_train) # fn = 8 and fp = 2639

#use model to predict validation dataset
predictions = gb_clf.predict(X_test)


# Display the results

# In[39]:


tn, fp, fn, tp = display_metrics(gb_clf, X_train, X_test, y_train, y_test, predictions, 'GB')
visualize(y_test, predictions, 'GB')
gb_auc = auc_roc_metrics(gb_clf, X_test, y_test, 'GB')
metrics_results['gb'] = gb_auc


# After tweaking the parameters, i can get a decent result from GradientBoostingClassifier. Changing the weights has a very large influence on the number of errors (FN and FP). Since this data is mostly 0 values, decreasing the weight of a true value vs a false value will decrease the FN, doing the opposite will decrease the FP. For one example run:  the sample_weight=np.where(y_train == 1,0.37,1.0) gives 13 FN and 795 FP. sample_weight=np.where(y_train == 1,0.1,1.0) gives 17 FN and 217 FP

# My next idea is to run 2 consecutive models consecutively. 1st model should have low false negatives to catch (almost) all the actual positives, even if the number of false positives is high. Then only take these records with a predicted 1 value (should only be a few thousand), as the input for the next model. 2nd test should have low false positives to weed out the actual negatives. Will use the Validate dataset on the 2 models created from the Train and Test datasets
# 
# Here are some details on the new model:
# 
# Current:
# Full Dataset -> Train -> Build M1(Train) -> Run M1(Test) -> Filter(Predicted 1's from Test) -> Build M2 -> run M2(Filtered Test)
#                 Test
#                 
# To Do:               
# Full Dataset -> Train -> Build M1(Train) -> Run M1(Test) -> Filter(Predicted 1's from Test) -> Build M2 -> run M1 and M2(Validate)
#                 Test
#                 Validate
# 
# Can also try the inverse, but think that option will have less chance of success.

# 1st step
# 
# build the 1st model to be used later on the validate dataset

# In[40]:


#setup model parameters, change some of the defaults based on benchmarking
gb_clf1 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=5, 
                                    max_depth=3, random_state=None, subsample = 1.0, criterion='mse', 
                                    min_samples_split = 10, min_samples_leaf = 10)

#default fit model
#gb_clf1.fit(X_train, y_train)

#since a false negative is much more likely than a false positive, we should weight them accordingly. 
#IE Finding a true one is more important, also more rare
gb_clf1.fit( X_train, y_train, sample_weight=np.where(y_train == 1,3.6,1.4) ) # was 5.0

#use model to predict validation dataset
predictions = gb_clf1.predict(X_test) 


# In[41]:


algo = 'GB1 Train **'
tn1, fp1, fn1, tp1 = display_metrics(gb_clf1, X_train, X_test, y_train, y_test, predictions, algo)
visualize(y_test, predictions, algo)
gb1_auc = auc_roc_metrics(gb_clf1, X_test, y_test, algo)
metrics_results['gb1_train'] = gb1_auc


# 2nd step takes all the Predicted Positives (the misclassified FP from upper right (~ 14000) plus the TP (since we won't use the actual value until the validation step)) and reprocesses these using a different model. The other 2 squares (Predicted 0's) are not included in the 2nd model, since we already have a low False negative result, so the initial predicted 0s don't change. Will need to add those back into the final results at the end.

# Add 1st model prediction column to X_test for filtering

# In[42]:


X_test['Prediction'] = predictions


# select rows with prediction of 1

# In[43]:


yes_ind = X_test[X_test['Prediction'] == 1].index


# Create 2nd train dataset from 1st dataset where the prediction was 1

# In[44]:


X2_test = X_test.loc[yes_ind]
y2_test = y_test.loc[yes_ind]


# clean up the X_test dataset for future modeling, means remove the Prediction column

# In[45]:


X_test = X_test.drop(['Prediction'], axis=1)
X2_test = X2_test.drop(['Prediction'], axis=1)


# Look at the prediction values from the first model (preda_1) for the rows with a predicted label of 0

# In[46]:


proba = gb_clf1.predict_proba(X2_test) 
pred = gb_clf1.predict(X2_test) 
df = pd.DataFrame(data=proba[:,0], columns=["preda_1"])
df.hist(bins=20, figsize=(10,5))
plt.show()


# Then we look at the ROC curve

# In[47]:


algo = 'PredictedPositives'
test_labels = y2_test
ns_probs = [0 for _ in range(len(test_labels))]
auc_roc_metrics_plots(proba[:,1], ns_probs, test_labels, algo)


# Next we build the 2nd model to be used model later on the validate dataset and look at the output

# In[48]:


#setup model parameters, change some of the defaults based on benchmarking
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=10, 
                                    max_depth=3, random_state=None, subsample = 1.0, criterion='mse', 
                                    min_samples_split = 10, min_samples_leaf = 10)

#default fit model
#gb_clf2.fit(X_train, y_train)

#since a false negative is much more likely than a false positive, we should weight them accordingly. 
#IE Finding a true one is more important
# note that the weights in the 2nd model are the inverse of the weights in the 1st model
gb_clf2.fit( X_train, y_train, sample_weight=np.where(y_train == 1,3.6,1.4) ) # was 0.1 but should be > 1 to work correctly

#use model to predict validation dataset
predictions = gb_clf2.predict(X2_test) 

algo = 'GB2 Train **'
tn, fp, fn, tp = display_metrics(gb_clf2, X_train, X2_test, y_train, y2_test, predictions, algo)

visualize(y2_test, predictions, algo)

gb2_auc = auc_roc_metrics(gb_clf2, X2_test, y2_test, algo)
metrics_results['gb2_train'] = gb2_auc

print("2 Step Final Confusion Matrix:")
print(tn+tn1, fp) 
print(fn+fn1, tp) 

fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names)

#create heatmap with combined data from both models
sns.heatmap(pd.DataFrame([[tn+tn1,fp],[fn+fn1,tp]]), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('2 Step Final Confusion matrix (Test)', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label')


# Now that we have built the 2 models from the test dataset, run the untouched validate dataset through both of them to get an unbiased result to compare against

# In[49]:


# run the validate dataset through the first model
algo = '2-Step'
predictions1 = gb_clf1.predict(X_val)
predictions_proba1 = gb_clf1.predict_proba(X_val)
X1_val_final = X_val.copy()
X1_val_final=X1_val_final.join(y_val)
X1_val_final['Proba_1'] = predictions_proba1[:,1]
#X1_val_final
#X_val = X_val.sort_index(axis = 0) 


# In[50]:


# adding this
# use both models to predict final validation dataset
algo = 'GB1 Validate **'
tn1, fp1, fn1, tp1 = display_metrics(gb_clf1, X_test, X_val, y_test, y_val, predictions1, algo) 
visualize(y_val, predictions1, algo)
gb1_auc = auc_roc_metrics(gb_clf1, X_val, y_val, algo)
metrics_results['gb1_validate'] = gb1_auc


# In[51]:



X_val['Prediction'] = predictions1

yes_ind = X_val[X_val['Prediction'] == 1].index

X2_val = X_val.loc[yes_ind]
y2_val = y_val.loc[yes_ind]
X2_val = X2_val.drop(['Prediction'], axis=1)
# run the validate dataset through the second model
predictions2 = gb_clf2.predict(X2_val)

X2_val_final = X2_val.copy()
X2_val_final.join(y2_val)
predictions_proba2 = gb_clf2.predict_proba(X2_val)
# validate the join!!
X2_val_final['Proba_2'] = predictions_proba2[:,1]
X2_val_final

cols_to_use = X2_val_final.columns.difference(X1_val_final.columns)
X_val_final = X1_val_final.join(X2_val_final[cols_to_use], how='left', lsuffix='_1', rsuffix='_2')
# rowwise action (axis=1)
X_val_final.loc[X_val_final['Proba_2'].isnull(),'Proba_2'] = X_val_final['Proba_1']
#X_val_final['Proba_2'].fillna(df['Proba_1'])
#X_val_final.query("Proba_1 != Proba_2")

#remove this column for use later
X_val = X_val.drop(['Prediction'], axis=1)


# In[52]:


algo = 'GB2 Validate **'
tn, fp, fn, tp = display_metrics(gb_clf2, X_train, X2_val, y_train, y2_val, predictions2, algo) 
visualize(y2_val, predictions2, algo)
gb2_auc = auc_roc_metrics(gb_clf2, X2_val, y2_val, algo)
metrics_results['gb2_validate'] = gb2_auc

print("2 Step Final Confusion Matrix:")
print(tn+tn1, fp) 
print(fn+fn1, tp) 

fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names)

#create heatmap with combined data from both models
sns.heatmap(pd.DataFrame([[tn+tn1,fp],[fn+fn1,tp]]), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('2 Step Final Confusion matrix (Validate)', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label')

algo = '2-Step'
Specificity = (tn+tn1)/(tn+tn1+fp)
Sensitivity = tp/(tp+fn+fn1)

print("Specificity =", Specificity)
print("Sensitivity =", Sensitivity)

print('2 Step Algorithm' + ':TEST | AUC Score: ' + str( round( (Specificity+Sensitivity)/2,3 )))

cm_results.append([algo, (tn+tn1), fp, (fn+fn1), tp])
# HERE
#two_step_auc = auc_roc_metrics(gb_clf, X_test, y_test, '2-Step')


# In[53]:


# try to combine the 2 models into one AUC score, however not sure that the proba values from 2 different models can be combined 

test_labels = X_val_final['Class']
ns_probs = [0 for _ in range(len(test_labels))]
model_probs = X_val_final['Proba_2']
model_pred=[1 if i > 0.50 else 0 for i in model_probs]

two_step_auc = auc_roc_metrics_plots(model_probs, ns_probs, test_labels, algo)

metrics_results['2-step'] = two_step_auc

cr_results.append([algo, classification_report(test_labels, model_pred)])


# In[54]:


y=np.reshape(test_labels.to_numpy(), -1)
fpr, tpr, thresholds = metrics.roc_curve(y, model_probs, pos_label=1)
roc.append([algo, fpr, tpr, thresholds])


# The 2 step process has the highest sensitivity (and specificity) between the models. The 2 step process also improves the overall model prediction of positives by a large amount (FP/TP ratio from above 10x to below 2x). I don't think we could get this high of precision and recall together with a single model. The best I could do with a single model was 10x FP/TP ratio.

# Next will try a few Neural Networks

# In[55]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
#from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras import backend as K
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras.layers import MaxPool1D
from keras.layers import Flatten
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation


# Adding swish activation function code for possible use later, can compare to relu, etc

# In[56]:


# create new activation function
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))


# In[57]:


# add this function to the list of Activation functions
get_custom_objects().update({'swish': Activation(swish)})


# Create the models to be used layer, using Sequential()

# In[58]:


def create_dnn(input_dim):
    # input_dim must equal number of features in X_train and X_test dataset
    clf1 = Sequential([
        Dense(units=16, kernel_initializer='uniform', input_dim=input_dim, activation='relu'),
        Dense(units=18, kernel_initializer='uniform', activation='relu'),
        Dropout(0.25),
        Dense(20, kernel_initializer='uniform', activation='relu'),
        Dense(24, kernel_initializer='uniform', activation='relu'),
        Dense(1, kernel_initializer='uniform', activation='sigmoid')
    ])
    return clf1


# In[59]:


def create_simple_dnn(input_dim):
    # input_dim must equal number of features in X_train and X_test dataset
    clf1 = Sequential([
        Dense(units=16, kernel_initializer='uniform', input_dim=input_dim, activation='relu'),
        Dense(units=18, kernel_initializer='uniform', activation='relu'),
        Dense(1, kernel_initializer='uniform', activation='sigmoid')
    ])
    return clf1


# In[60]:


def create_complex_dnn(input_dim):
    # input_dim must equal number of features in X_train and X_test dataset
    clf1 = Sequential([
        Dense(units=16, kernel_initializer='uniform', input_dim=input_dim, activation='relu'),
        Dense(units=18, kernel_initializer='uniform', activation='relu'),
        Dropout(0.10),
        Dense(units=30, kernel_initializer='uniform', activation='relu'),
        Dense(units=28, kernel_initializer='uniform', activation='relu'),
        Dropout(0.10),
        Dense(units=30, kernel_initializer='uniform', activation='relu'),
        Dense(units=28, kernel_initializer='uniform', activation='relu'),
        Dropout(0.10),
        Dense(units=20, kernel_initializer='uniform', activation='relu'),
        Dense(units=24, kernel_initializer='uniform', activation='relu'),
        Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
    ])
    return clf1


# In[61]:


def create_cnn(input_shape):
    model = Sequential()
    #model.add(Conv1D(32, 2, activation = 'relu', input_shape = input_shape))
    #model.add(Conv1D(filters=32, kernel_size=2, input_shape = (30) ))
    #model.add(Conv1D(filters=32, kernel_size=10, strides=1, activation='swish', padding='valid', input_shape=input_shape ))
    model.add(Conv1D(filters=32, kernel_size=10, strides=1, activation='relu', padding='valid', input_shape=input_shape ))
    # TypeError: 'int' object is not iterable
    model.add(BatchNormalization())
    model.add(MaxPool1D(2))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


# run the CNN model

# In[62]:


input_shape = (X_train.shape[1], 1)
input_dim = X_train.shape[1]
print("Input shape:", input_shape)
clf = create_cnn(input_shape)
# NameError: name 'input_shape' is not defined

# reshape data for CNN expected input
nrows, ncols = X_train.shape # (602,30)
X_train_arr = X_train.copy().to_numpy()
y_train_arr = y_train.copy()
X_train_arr = X_train_arr.reshape(nrows, ncols, 1)

nrows, ncols = X_test.shape # (602,30)
X_test_arr = X_test.copy().to_numpy()
y_test_arr = y_test.copy()
X_test_arr = X_test_arr.reshape(nrows, ncols, 1)

#opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
# Let's train the model using RMSprop
#clf.compile(loss='binary_crossentropy',
#              optimizer=opt,
#              metrics=['accuracy'])
# or
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

clf.summary()

#adam = keras.optimizers.Adam(learning_rate=0.001)
# try using focal_loss to give heavier weight to examples that are difficult to classify
# seems to improve the metrics slightly
#clf.compile(optimizer=adam, loss=[focal_loss], metrics=['accuracy'])

# create/fit model on the training dataset
#clf.fit(X_train, y_train, batch_size=16, epochs=32, sample_weight=np.where(y_train == 1,0.2,1.0).flatten())
#clf.fit(X_train, y_train, batch_size=16, epochs=20, sample_weight=np.where(y_train == 1,1.0,1.0).flatten())
# or
clf.fit(X_train_arr, y_train_arr, epochs=200, verbose=verbose, sample_weight=np.where(y_train_arr == 1,1.0,1.0).flatten())
# check model metrics
score = clf.evaluate(X_train_arr, y_train_arr, batch_size=128)
print('\nAnd the Train Score is ', score[1] * 100, '%')
score = clf.evaluate(X_test_arr, y_test_arr, batch_size=128)
print('\nAnd the Test Score is ', score[1] * 100, '%')
# predict probabilities for test set
yhat_probs = clf.predict(X_test_arr, verbose=verbose)
# predict crisp classes for test set
yhat_classes = clf.predict_classes(X_test_arr, verbose=verbose)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
print("Classification Report (CNN)") 
print(classification_report(y_test_arr, yhat_classes))

tn, fp, fn, tp = display_metrics(clf, X_train_arr, X_test_arr, y_train_arr, y_test_arr, yhat_classes, 'CNN')
visualize(y_test_arr, yhat_classes, 'CNN')
cnn_auc = auc_roc_metrics(clf, X_test_arr, y_test_arr, 'CNN')
metrics_results['cnn'] = cnn_auc


# In[63]:


X_train.shape[1]


# Now run the basic DNN (Deep Neural Network)

# In[64]:


clf = create_dnn(input_dim)
clf.summary()
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#adam = keras.optimizers.Adam(learning_rate=0.001)
# try using focal_loss to give heavier weight to examples that are difficult to classify
# seems to improve the metrics slightly
#clf.compile(optimizer=adam, loss=[focal_loss], metrics=['accuracy'])

# create/fit model on the training dataset
#clf.fit(X_train, y_train, batch_size=16, epochs=32, sample_weight=np.where(y_train == 1,0.2,1.0).flatten())
clf.fit(X_train, y_train, batch_size=16, epochs=20, verbose=verbose, sample_weight=np.where(y_train == 1,1.0,1.0).flatten())

# check model metrics
score = clf.evaluate(X_train, y_train, batch_size=128)
print('\nAnd the Train Score is ', score[1] * 100, '%')
score = clf.evaluate(X_test, y_test, batch_size=128)
print('\nAnd the Test Score is ', score[1] * 100, '%')
# predict probabilities for test set
yhat_probs = clf.predict(X_test, verbose=verbose)
# predict crisp classes for test set
yhat_classes = clf.predict_classes(X_test, verbose=verbose)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
print("Classification Report (DNN)") 
print(classification_report(y_test, yhat_classes))

tn, fp, fn, tp = display_metrics(clf, X_train, X_test, y_train, y_test, yhat_classes, 'DNN')
visualize(y_test, yhat_classes, 'DNN')
dnn_auc = auc_roc_metrics(clf, X_test, y_test, 'DNN')
metrics_results['dnn'] = dnn_auc


# Results from Deep NN are better than 1 step/model examples, but overall not quite as good as the 2 step/model process. I can get the sensitivity to be as good, but in that case, the specificity is much lower. As more data is added or processed through this DNN, the results should improve, maybe eventually beating the 2 step model. However, it seems that increasing the number of epochs will weight the model to higher false negatives, similar to using sample weights for the GBM model:
# 
# **sample_weight=np.where(y_train == 1,0.1,1.0)**
# 
# **giving a 1 in the training data 10 times the weight or inflence of a 0**
# 
# For now, we will keep the number of epochs at 5.
# Weighting has the same effect on this DNN as it had on the GBM. Best all around result with 
# 
# sample_weight=np.where(y_train == 1,0.1,1.0).flatten()

# Look at simpler and more complex examples of a DNN for comparison

# In[65]:


clf = create_simple_dnn(input_dim)
clf.summary()
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# create/fit model on the training dataset
#clf.fit(X_train, y_train, batch_size=15, epochs=5, sample_weight=np.where(y_train == 1,0.1,1.0).flatten())
clf.fit(X_train, y_train, batch_size=32, epochs=32, verbose=verbose, sample_weight=np.where(y_train == 1,1.0,1.0).flatten())
#clf.fit(X_train, y_train, batch_size=15, epochs=5, sample_weight=np.where(y_train == 1,5.0,1.0).flatten())
#clf.fit(X_train, y_train, batch_size=15, epochs=5)

# check model metrics
score = clf.evaluate(X_train, y_train, batch_size=128)
print('\nAnd the Train Score is ', score[1] * 100, '%')
score = clf.evaluate(X_test, y_test, batch_size=128)
print('\nAnd the Test Score is ', score[1] * 100, '%')
# predict probabilities for test set
yhat_probs = clf.predict(X_test, verbose=verbose)
# predict crisp classes for test set
yhat_classes = clf.predict_classes(X_test, verbose=verbose)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
print("Classification Report (DNN Simple)") 
print(classification_report(y_test, yhat_classes))
tn, fp, fn, tp = display_metrics(clf, X_train, X_test, y_train, y_test, yhat_classes, 'DNN Simple')
visualize(y_test, yhat_classes, 'DNN Simple')
dnn_simple_auc = auc_roc_metrics(clf, X_test, y_test, 'DNN-Simple')
metrics_results['dnn_simple'] = dnn_simple_auc


# This DNN is successful at reducing the FP/TP ratio. This is expected as a Neural Network can decide on its own rules to include based on the input data. Below I try other more and less complex methods, but so far the results are not as good.

# In[66]:


clf = create_complex_dnn(input_dim)
clf.summary()
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# create/fit model on the training dataset
#clf.fit(X_train, y_train, batch_size=15, epochs=5, sample_weight=np.where(y_train == 1,0.1,1.0).flatten())
clf.fit(X_train, y_train, batch_size=16, epochs=32, verbose=verbose, sample_weight=np.where(y_train == 1,4.0,1.0).flatten())
#clf.fit(X_train, y_train, batch_size=15, epochs=5, sample_weight=np.where(y_train == 1,5.0,1.0).flatten())
#clf.fit(X_train, y_train, batch_size=15, epochs=5)

# check model metrics
score = clf.evaluate(X_train, y_train, batch_size=128)
print('\nAnd the Train Score is ', score[1] * 100, '%')
score = clf.evaluate(X_test, y_test, batch_size=128)
print('\nAnd the Test Score is ', score[1] * 100, '%')
# predict probabilities for test set
yhat_probs = clf.predict(X_test, verbose=verbose)
# predict crisp classes for test set
yhat_classes = clf.predict_classes(X_test, verbose=verbose)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
print("Classification Report (DNN complex)") 
print(classification_report(y_test, yhat_classes))
tn, fp, fn, tp = display_metrics(clf, X_train, X_test, y_train, y_test, yhat_classes, 'DNN Complex')
visualize(y_test, yhat_classes, 'DNN Complex')
dnn_complex_auc = auc_roc_metrics(clf, X_test, y_test, 'DNN-Complex')
metrics_results['dnn_complex'] = dnn_complex_auc


# In[67]:


def create_autoencoder(input_dim):
    # input_dim must equal number of features in X_train and X_test dataset
    clf1 = Sequential([
        Dense(units=15, kernel_initializer='uniform', input_dim=input_dim, activation='tanh', activity_regularizer=regularizers.l1(10e-5)),
        Dense(units=7, kernel_initializer='uniform', activation='relu'),
        Dense(units=7, kernel_initializer='uniform', activation='tanh'),
        Dense(units=31, kernel_initializer='uniform', activation='relu'),
        Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
    ])
    return clf1


# In[68]:


clf = create_autoencoder(input_dim)
clf.summary()
#clf.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# create/fit model on the training dataset
#clf.fit(X_train, y_train, batch_size=32, epochs=32, shuffle=True)#, validation_data=(X_test, X_test))
clf.fit(X_train, y_train, batch_size=16, epochs=32, verbose=verbose, sample_weight=np.where(y_train == 1,2.0,1.0).flatten())
#clf.fit(X_train, y_train, batch_size=32, epochs=32, sample_weight=np.where(y_train == 1,0.1,1.0).flatten())
#clf.fit(X_train, y_train, batch_size=15, epochs=5)

# check model metrics
score = clf.evaluate(X_train, y_train, batch_size=32)
print('\nAnd the Train Score is ', score[1] * 100, '%')
score = clf.evaluate(X_test, y_test, batch_size=32)
print('\nAnd the Test Score is ', score[1] * 100, '%')
# predict probabilities for test set
yhat_probs = clf.predict(X_test, verbose=verbose)
# predict crisp classes for test set
yhat_classes = clf.predict_classes(X_test, verbose=verbose)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
print("Classification Report (AutoEncoder)") 
print(classification_report(y_test, yhat_classes))
tn, fp, fn, tp = display_metrics(clf, X_train, X_test, y_train, y_test, yhat_classes, 'AutoEncoder')
visualize(y_test, yhat_classes, 'AutoEncoder')
autoencoder_auc = auc_roc_metrics(clf, X_test, y_test, 'AutoEncoder')
metrics_results['autoencoder'] = autoencoder_auc


# In[69]:


print("AUC comparisons")
print(metrics_results)


# <pre>
# AUC comparisons between all the models:
# 
# {'lr': 0.965, 'rf': 0.975, 'gb': 0.975, 'gb1_train': 0.979, 'gb2_train': 0.967, 'gb1_validate': 0.99, 'gb2_validate': 0.974, '2-step': 0.941, 'dnn': 0.964, 'dnn_simple': 0.978, 'dnn_complex': 0.939, 'autoencoder': 0.956}
# {'lr': 0.968, 'rf': 0.979, 'gb': 0.976, 'gb1_train': 0.975, 'gb2_train': 0.968, 'gb1_validate': 0.991, 'gb2_validate': 0.978, '2-step': 0.957, 'dnn': 0.983, 'dnn_simple': 0.981, 'dnn_complex': 0.961, 'autoencoder': 0.952}
# 
# Side by Side comparisions of all models
# 
# 
#     LR
#     [[64954  3280]
#      [   15   104]]
# 
#     Classification Report
#                   precision    recall  f1-score   support
# 
#                0       1.00      0.95      0.98     68234
#                1       0.03      0.87      0.06       119
# 
#         accuracy                           0.95     68353
#        macro avg       0.52      0.91      0.52     68353
#     weighted avg       1.00      0.95      0.97     68353
# 
# 
#     RF
#     [[66254  1980]
#      [   13   106]]
# 
#     Classification Report
#                   precision    recall  f1-score   support
# 
#                0       1.00      0.97      0.99     68234
#                1       0.05      0.89      0.10       119
# 
#         accuracy                           0.97     68353
#        macro avg       0.53      0.93      0.54     68353
#     weighted avg       1.00      0.97      0.98     68353
# 
# 
#     GB
#     [[66732  1502]
#      [   16   103]]
#     Classification Report
#                   precision    recall  f1-score   support
# 
#                0       1.00      0.98      0.99     68234
#                1       0.06      0.87      0.12       119
# 
#         accuracy                           0.98     68353
#        macro avg       0.53      0.92      0.55     68353
#     weighted avg       1.00      0.98      0.99     68353
# 
# 
#     2Step
#     [[45336  162]
#      [   5    67]]
# 
#     Classification Report
#                   precision    recall  f1-score   support
# 
#                0       1.00      0.99      0.99     45377
#                1       0.36      0.93      0.52        72
#         accuracy                           0.??     45449
#        macro avg       0.57      0.96      0.71     45449
#     weighted avg       1.00      0.99      0.99     45449
# 
# 
#     DNN
#     [[64991  3243]
#      [    9   110]]
# 
#     Classification Report
#                   precision    recall  f1-score   support
# 
#                0       1.00      0.95      0.98     68234
#                1       0.03      0.92      0.06       119
# 
#         accuracy                           0.95     68353
#        macro avg       0.52      0.94      0.52     68353
#     weighted avg       1.00      0.95      0.97     68353
# 
# 
#     DNN Simple
#     [[63011  5223]
#      [    6   113]]
# 
#     Classification Report
#                   precision    recall  f1-score   support
# 
#                0       1.00      0.92      0.96     68234
#                1       0.02      0.95      0.04       119
# 
#         accuracy                           0.92     68353
#        macro avg       0.51      0.94      0.50     68353
#     weighted avg       1.00      0.92      0.96     68353
# 
# 
#     AutoEncoder
#     60277 7957 7 112
#     [[60277  7957]
#      [    7   112]]
#     Classification Report
#                   precision    recall  f1-score   support
# 
#                0       1.00      0.88      0.94     68234
#                1       0.01      0.94      0.03       119
# 
#         accuracy                           0.88     68353
#        macro avg       0.51      0.91      0.48     68353
#     weighted avg       1.00      0.88      0.94     68353
#     
#     
#     CNN
#     [[64761  3473]
#      [   14   105]]
#     Classification Report
#                   precision    recall  f1-score   support
# 
#                0       1.00      0.95      0.97     68234
#                1       0.03      0.88      0.06       119
# 
#         accuracy                           0.95     68353
#        macro avg       0.51      0.92      0.52     68353
#     weighted avg       1.00      0.95      0.97     68353
# 
# 
# 

# In[70]:


plt.figure(figsize=(7,5),dpi=100)

for i in range(0,len(roc)):
    #print('roc[0]', roc[0])
    #print('roc[i]', roc[i])
    auc1 = auc(roc[i][1],roc[i][2])
    plt.plot(roc[i][1],roc[i][2], label="AUC {0}:{1}".format(roc[i][0], auc1), linewidth=2)
    
plt.plot([0, 1], [0, 1], 'k--', lw=1) 
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate') 
plt.title('ROC') 
plt.grid(True)
plt.legend(loc="lower right")
plt.show()


# Final confusion matrix results comparing the different algorithms. The items marked with ** are interim results for the 2 step process, and are not for comparison, only shown for reference. As you can see, both the FP and FN values are best for the 2 step process. This process is the most efficient at finding fraudulent transactions, and has the least amount of noise (FP).

# number of Actual 0 and 1 in the final validation dataset for 2-test model
# "1" total should match the FN + TP

# In[71]:


y_val['Class'].value_counts()


# number of Actual 0 and 1 in the final test dataset for all other models
# "1" total should match the FN + TP

# In[72]:


y_test['Class'].value_counts()


# Here are the final results in tabular form. 

# In[73]:


final_results = pd.DataFrame(cm_results, columns=('algo','TN','FP','FN','TP')) 
#sp = round((tn1 + tn2)/(tn1 + tn2 +fp2), 3)
#se = round(tp2/(tp2 + fn1 + fn2), 3)
final_results['SP'] = round(final_results['TN']/(final_results['TN'] + final_results['FP']), 3)
final_results['SE'] = round(final_results['TP']/(final_results['TP'] + final_results['FN']), 3)
final_results['Avg'] = (final_results['SP'] + final_results['SE'])/2
print('test, val, split settings')
print(spl1,spl2)
print('test, val, split sizes')
print( (spl1-spl1*spl2), (spl1*spl2) )
filtered = final_results[~final_results.algo.str.contains('a', regex= True, na=False)]
sort = filtered.sort_values(filtered.columns[7], ascending = False) 
print(sort)
sort.to_csv('results.csv', sep=',', mode='a', encoding='utf-8', header=True)


# In[ ]:




