#!/usr/bin/env python
# coding: utf-8

# <h1 align='center' style='color:purple'>Credit Card Fraud - Imbalanced Data Set</h1>

# Winning kaggle notebook: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284  
# https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600  
# 
# Try some of these ideas: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data  
# 
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

import datetime

import random as rn
rn.seed(1) # random
import numpy as np
#from numpy.random import seed
np.random.seed(7) # or 7
import tensorflow as tf
tf.random.set_seed(0) # tf

import pandas as pd
import os
import tempfile

import matplotlib as mpl                                                                                             
if os.environ.get('DISPLAY','') == '':                                                                               
    print('no display found. Using non-interactive Agg backend')                                                     
    mpl.use('Agg')                                                                    
    
import matplotlib.pyplot as plt
if (os.environ.get('TERM','') == 'xterm-color'): 
    get_ipython().run_line_magic('matplotlib', 'inline')
elif (os.environ.get('TERM','') == 'cygwin'):
    print("shell terminal found")
else: # 'cygwin'
    print("no terminal found")
    
import pandas_profiling as pp
import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew, kurtosis, boxcox #for some statistics

from scipy.special import boxcox1p, inv_boxcox, inv_boxcox1p

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
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
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, Normalizer
from matplotlib import pyplot
import zipfile
import time

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
from keras.optimizers import Adam, SGD, RMSprop

StartTime = datetime.datetime.now()


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


#print(os.environ)
print('TERM:', os.environ.get('TERM',''))


# Always like to include a timer function to see where my code is running slow or taking most of the run time

# In[4]:


#import keras.backend as K
def custom_loss(y_true, y_pred):
    #loss = abs(y_true - y_pred)                                                                                   
    mask1 = K.less(y_pred, y_true) # is y_pred < y_true or y_pred - y_true < 0, FN                                 
    mask2 = K.less(y_true, y_pred) # is y_true < y_pred or y_true - y_pred < 0, FP                                 
    #loss = K.cast(mask1, K.floatx()) * 2 * (y_true - y_pred) # only include FN                                    
    loss = (K.cast(mask1, K.floatx()) * 2 * (y_true - y_pred)) + (K.cast(mask2, K.floatx()) * 4 * (y_pred - y_true) ) # FP has higher penalty                                                                                          
    return loss 

def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def custom_loss_function(y_true, y_pred):
    squared_difference = K.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)

def specificity(y_pred, y_true):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels 
    Returns:
    Specificity score
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity


# In[5]:


# define variable learning rate function
from keras.callbacks import LearningRateScheduler, EarlyStopping, History, LambdaCallback
import math

def step_decay(epoch, lr):
    drop = 0.999 # was .999
    epochs_drop = 50.0 # was 175, sgd likes 200+, adam likes 100
    lrate = lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    #print("epoch=" + str(epoch) + " lr=" + str(lr) + " lrate=" + str(lrate))
    return lrate

lrate = LearningRateScheduler(step_decay)
early_stopping = EarlyStopping(monitor='val_loss', patience=25, mode='auto', restore_best_weights = True)
callbacks_list = [lrate, early_stopping] 


# In[6]:


# list of all metrics: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.BinaryCrossentropy(name='crossentropy', from_logits=False, label_smoothing=0),
      keras.metrics.SensitivityAtSpecificity(specificity=0.0, num_thresholds=200, name='sensitivity'),  # (tp / (tp + fn)).
      keras.metrics.SpecificityAtSensitivity(sensitivity=0.0, num_thresholds=200, name='specificity'),   # (tn / (tn + fp)).
      keras.metrics.CosineSimilarity(name='cosine_similarity', axis=-1),
      keras.metrics.KLDivergence(name='kl_divergence'),
      keras.metrics.LogCoshError(name='logcosh'),
]

BATCH_SIZE = 2048
EPOCHS = 100


# In[7]:


def make_model(metrics = METRICS, output_bias=-100, opt_sel='adam'):
    #if output_bias is not None:
    print('func_output_bias:', output_bias)
    
    METRICS = [
          keras.metrics.TruePositives(name='tp'),
          keras.metrics.FalsePositives(name='fp'),
          keras.metrics.TrueNegatives(name='tn'),
          keras.metrics.FalseNegatives(name='fn'), 
          keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
          keras.metrics.Precision(name='precision'),
          keras.metrics.Recall(name='recall'),
          keras.metrics.AUC(name='auc'),
          #keras.metrics.BinaryCrossentropy(name='crossentropy', from_logits=False, label_smoothing=0),
          keras.metrics.BinaryCrossentropy(name='crossentropy'),
          keras.metrics.SensitivityAtSpecificity(specificity=0.0, num_thresholds=200, name='sensitivity'),  # (tp / (tp + fn)).
          keras.metrics.SpecificityAtSensitivity(sensitivity=0.0, num_thresholds=200, name='specificity'),   # (tn / (tn + fp)).
          keras.metrics.CosineSimilarity(name='cosine_similarity', axis=-1),
          keras.metrics.KLDivergence(name='kl_divergence'),
          #custom_loss_mask,
          #custom_loss_function
    ]

    if (output_bias > -50):
        print("output_bias passed in:", output_bias)
        use_bias_sel = True
        output_bias = keras.initializers.Constant(output_bias)
        clf = Sequential([
                Dense(units=16, activation='relu', input_shape=(train_features.shape[-1],)),
                Dropout(0.50),
                Dense(units=1, activation='sigmoid', use_bias=use_bias_sel, bias_initializer=output_bias),
            ])
    else:
        print("output_bias not passed in")
        use_bias_sel = False
        clf = Sequential([
                Dense(units=16, activation='relu', input_shape=(train_features.shape[-1],)),
                Dropout(0.50),
                Dense(units=1, activation='sigmoid'),
            ])

    learning_rate = 0.001
    decay = 0.0002
    momentum=0.99
    #opt_sel = "adam"
    if (opt_sel == "adam"):
        #opt = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad) # added to v86
        opt = Adam(lr=learning_rate)
    elif(opt_sel == "sgd"):
        opt = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay, nesterov=True)
    
    clf.compile(
        #optimizer=keras.optimizers.Adam(lr=1e-3),
        optimizer=opt,
        loss=keras.losses.BinaryCrossentropy(),
        #loss=custom_loss_mask,
        #loss=custom_loss_function,
        metrics=METRICS)

    return clf;


# In[8]:


def CalcPct(df,title):
    unique_elements, counts_elements = np.unique(df, return_counts=True)
    calc_pct = round(counts_elements[1]/(counts_elements[0]+counts_elements[1]) * 100,6)
    print(title)
    print(np.asarray((unique_elements, counts_elements)))
    return calc_pct


# In[9]:


colab = os.environ.get('COLAB_GPU', '10')
if (int(colab) == 0):
    from google.colab import drive
    drive.mount('/content/drive')  
else:
    print("colab_gpu not found")


# Setup to run on Google Colab and Kaggle platforms

# In[10]:


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


# In[11]:


verbose=0
# Load the Data Set
df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
#off line data source for backup
#df = pd.read_csv('creditcard.csv')


# Public Credit Card Dataset. This is financial data, and is considered to be sensitive so it is "encrypted" through the use of PCA to protect privacy. Only the Time and Dollar columns are intact after the "encryption"
# 
# Doing some initial data exploration

# In[12]:


# Check the data, make sure it loaded okay
print(df.head())


# In[13]:


# Check the datatypes of the Data set 
df.info()


# In[14]:


# Check the Uniqueness
df.nunique()


# In[15]:


# Check for missing data
df.isnull().sum()


# In[16]:


# Check basic Statistics
# looks like StandardScaler was performed on this dataset, mean is close to 0 for all columns

df.describe(include ='all')


# In[17]:


# Check the Class Imbalance of the Data 

df['Class'].value_counts()


# In[18]:


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

# In[19]:


f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(method='kendall'), fignum=f.number) # pearson or spearman are also available
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title("Kendall'sCorrelation Matrix Full Data Set", fontsize=16)


# V21 and V22 show the highest tau-b score, will investigate this relationship later

# In[20]:


#try some data cleansing, Amount has a few high values, so try using the log of that column instead.

temp_df = df.copy()
temp_df = temp_df.drop(['Time'], axis=1)
temp_df['Log_Amount'] = np.log(temp_df.pop('Amount')+0.001)
df = temp_df.copy()


# In[21]:


from scipy.special import boxcox1p
lam = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,2,3]
for i in lam:
    print("lam:", i)
    X = df.loc[:, df.columns != 'Class']
    y = df.loc[:, df.columns == 'Class']
    norm = MinMaxScaler().fit(X)
    X = pd.DataFrame(norm.transform(X), index=X.index, columns=X.columns)
    numeric_feats = X.dtypes[X.dtypes != "object"].index
    skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness = skewness[abs(skewness) > 2]
    skewness = skewness[skewness.Skew == skewness.Skew]
    print("Pre: There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
    print("Pre", abs(skewness.Skew).mean())
    skewed_features = skewness.index
    lam_f = 0.0
    for feat in skewed_features:
        X[feat] = boxcox1p(X[feat], i)
    numeric_feats = X.dtypes[X.dtypes != "object"].index
    skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness = skewness[abs(skewness) > 2]
    skewness = skewness[skewness.Skew == skewness.Skew]
    print("Post: There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
    print("Post", abs(skewness.Skew).mean())


# In[22]:


from scipy.special import boxcox1p
X = df.loc[:, df.columns != 'Class']
y = df.loc[:, df.columns == 'Class']
numeric_feats = X.dtypes[X.dtypes != "object"].index
skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness) > 1.0]
skewness = skewness[skewness.Skew == skewness.Skew]
print("Pre: There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
print("Pre", abs(skewness.Skew).mean())
skewed_features = skewness.index
pt = PowerTransformer(method='yeo-johnson').fit(X)
X = pd.DataFrame(pt.transform(X), index=X.index, columns=X.columns)
numeric_feats = X.dtypes[X.dtypes != "object"].index
skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness) > 1.0]
skewness = skewness[skewness.Skew == skewness.Skew]
print("Post: There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
print("Post", abs(skewness.Skew).mean())


# In[23]:


X.hist(bins=20, figsize=(20,15))
plt.show()


# In[24]:


numeric_feats = X.dtypes[X.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# Need to normalize the data before using boxcox or log transforms as they don't work on negative and 0 values

# In[25]:


X = df.loc[:, df.columns != 'Class']
y = df.loc[:, df.columns == 'Class']
norm = MinMaxScaler().fit(X)
X = pd.DataFrame(norm.transform(X), index=X.index, columns=X.columns)


# Pre-transform skew

# In[26]:


numeric_feats = X.dtypes[X.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in all numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[27]:


skewness = skewness[abs(skewness) > 0.75]
skewness = skewness[skewness.Skew == skewness.Skew]
print("There are {} highly skewed numerical features to Box Cox transform".format(skewness.shape[0]))


# In[28]:


skewness


# In[29]:


abs(skewness.Skew).mean()


# In[30]:


from scipy.special import boxcox1p
skewed_features = skewness.index

lam_f = 0.0
for feat in skewed_features:
    X[feat] = boxcox1p(X[feat], lam_f)


# Post-transform skew

# In[31]:


numeric_feats = X.dtypes[X.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[32]:


skewness = skewness[abs(skewness) > 0.75]
skewness = skewness[skewness.Skew == skewness.Skew]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))


# In[33]:


skewness


# In[34]:


abs(skewness.Skew).mean()


# In[35]:


X.hist(bins=20, figsize=(20,15))
plt.show()


# so far the MinMaxScaler, boxcox1p and log1p transforms make the data more skewed...
# just utilize the PowerTransformer instead, with yeo-johnson as there are many negative values

# Divide the dataset into features and labels and then into Train, Test and Validate datasets

# In[36]:


X = df.loc[:, df.columns != 'Class']
y = df.loc[:, df.columns == 'Class']

# divide full data into features and label
spl1 = 0.36
spl2 = 0.44

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
y_train_orig = y_train.copy(deep=True)


# In[37]:


type(y_test)


# The correct way to transform, fit the train data and transform train, test and val data based on the fit  
# This does not have any effect on the performance of the model. Mean Specificity and Sensitivity are unchanged. Tested ~ 20 iterations

# In[38]:


pt = PowerTransformer(method='yeo-johnson', standardize=True).fit(X_train)
X_train_pt = pd.DataFrame(pt.transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_pt  = pd.DataFrame(pt.transform(X_test), index=X_test.index, columns=X_test.columns)
X_val_pt   = pd.DataFrame(pt.transform(X_val), index=X_val.index, columns=X_val.columns)

sc = StandardScaler().fit(X_train)
X_train_sc = pd.DataFrame(sc.transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_sc  = pd.DataFrame(sc.transform(X_test), index=X_test.index, columns=X_test.columns)
X_val_sc   = pd.DataFrame(sc.transform(X_val), index=X_val.index, columns=X_val.columns)    


# In[39]:


f = plt.figure(figsize=(16, 12))
plt.matshow(X_train.corr(method='kendall'), fignum=f.number) # pearson or spearman are also available
plt.xticks(range(X_train.shape[1]), X_train.columns, fontsize=14, rotation=45)
plt.yticks(range(X_train.shape[1]), X_train.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title("Kendall's Correlation Matrix Initial Train Set", fontsize=16)
plt.show()


# In[40]:


# prepare data for model, need to do this normalization and clipping separately for X_train, X_test and X_val 
# to avoid any contamination between Train and Test/Validate datasets

# Use a utility from sklearn to split and shuffle our dataset.
train_df, test_df = train_test_split(df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

save_data = 1
if (save_data == 1):
    df1 = train_df.copy()
    df1.to_csv('CCFraudTrain.csv', index=False)  
    df2 = test_df.copy()
    df2.to_csv('CCFraudTest.csv', index=False) 
    df3 = val_df.copy()
    df3.to_csv('CCFraudVal.csv', index=False) 


# In[41]:


train_df


# In[42]:


if (save_data == 1):
    train_df = pd.read_csv('C:\DataScience\Repo\Imbalanced_data\CreditCardFraud\working\Imb_Train.csv')                                                                         
    test_df = pd.read_csv('C:\DataScience\Repo\Imbalanced_data\CreditCardFraud\working\Imb_Test.csv')  
    val_df = pd.read_csv('C:\DataScience\Repo\Imbalanced_data\CreditCardFraud\working\Imb_Val.csv')     


# In[43]:


# Form np arrays of labels and features.
train_labels_df = train_df[['Class']]
test_labels_df = test_df[['Class']]
val_labels_df = val_df[['Class']]

train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

scaler = StandardScaler()

train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)

ss = StandardScaler().fit(train_df)
train_features_df = pd.DataFrame(ss.transform(train_df), index=train_df.index, columns=train_df.columns)
test_features_df = pd.DataFrame(ss.transform(test_df), index=test_df.index, columns=test_df.columns)
val_features_df = pd.DataFrame(ss.transform(val_df), index=val_df.index, columns=val_df.columns)

train_features_df = pd.DataFrame(np.clip(train_features_df, -5, 5), index=train_features_df.index, columns=train_features_df.columns)
test_features_df = pd.DataFrame(np.clip(test_features_df, -5, 5), index=test_features_df.index, columns=test_features_df.columns)
val_features_df = pd.DataFrame(np.clip(val_features_df, -5, 5), index=val_features_df.index, columns=val_features_df.columns)

print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)


# In[44]:


class_names=[0,1] # name  of classes 1=fraudulent transaction

y_val['Class'].value_counts()

TrainPct = CalcPct(y_train,"Train")
TestPct = CalcPct(y_test,"Test")
ValPct = CalcPct(y_val,"Val")
zeros, ones = np.bincount(y_train['Class'])


# Investigate the high tau-b value between V21 and V22

# In[45]:


# Form np arrays of labels and features for jointplot charts

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

# In[46]:


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
X_train_ds = X_train.loc[undersample_ind]
y_train_ds = y_train.loc[undersample_ind]

y_train_ds = np.array(y_train_ds).flatten()
y_train = np.array(y_train).flatten()


# Create some calculation and visualization functions to show the results

# In[47]:


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


# In[48]:


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
    print('SS Avg =', ( (tn/(tn+fp)) + (tp/(tp+fn)))/2)
    if (type(test_label) != np.ndarray):
        y = np.reshape(test_label.to_numpy(), -1)
    else:
        y = test_label
    fpr, tpr, thresholds = metrics.roc_curve(y, model_probs, pos_label=1)
    cm_results.append([algo, tn, fp, fn, tp])
    cr_results.append([algo, classification_report(test_label, pred)])
    roc.append([algo, fpr, tpr, thresholds])
    # AUC score should be (Sensitivity+Specificity)/2
    print(algo + ':TEST | AUC Score: ' + str( round(metrics.auc(fpr, tpr),3 )))
    return tn, fp, fn, tp


# In[49]:


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


# In[50]:


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


# In[51]:


# Define our custom loss function
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


# In[52]:


def prediction_cutoff(model, test_features, cutoff):
    model.predict_proba(test_features)
    # to get the probability in each class, 
    # for example, first column is probability of y=0 and second column is probability of y=1.

    # the probability of being y=1
    prob1=model.predict_proba(test_features)[:,1]
    predicted=[1 if i > cutoff else 0 for i in prob1]
    return predicted


# In[53]:


metrics_results = {}
roc = []
cm_results = []
cr_results = []


# run Logistic Regression model first

# In[54]:


neg, pos = np.bincount(y_train)
total = neg + pos
initial_bias = np.log([pos/neg])

weight_for_0 = (1 / neg)*(total)/2.0 # was 5
weight_for_1 = (1 / pos)*(total)/2.0 # was 2


# In[55]:


lr = LogisticRegression()
#lr = LogisticRegression(solver='lbfgs')

lr.fit(X_train_ds, y_train_ds)
#lr.fit(X_train_ds, y_train_ds, sample_weight=np.where(y_train_ds == 1, weight_for_1, weight_for_0).flatten())
#lr_Pred = lr.predict(X_test)
# or
lr_Pred = prediction_cutoff(lr, X_test, 0.5) # 0.5 is the default cutoff for a logistic regression test
print(metrics.accuracy_score(y_test, lr_Pred))
tn, fp, fn, tp = display_metrics(lr, X_train_ds, X_test, y_train_ds, y_test, lr_Pred, 'LR')
visualize(y_test, lr_Pred, 'LR') # actual labels vs predicted labels
lr_auc = auc_roc_metrics(lr, X_test, y_test, 'LR')
metrics_results['lr'] = lr_auc


# Show the results of this model

# In[56]:


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

# In[57]:


rf = RandomForestClassifier(n_estimators = 1000)
rf.fit(X_train_ds, y_train_ds)

#rf = RandomForestClassifier(n_estimators = 100, class_weight={ 1: weight_for_1, 0: weight_for_0})
#rf.fit(X_train, y_train, sample_weight=np.where(y_train == 1, weight_for_1, weight_for_0).flatten())

rf_Pred=rf.predict(X_test)

#print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, rf_Pred))
tn, fp, fn, tp = display_metrics(rf, X_train_ds, X_test, y_train_ds, y_test, rf_Pred, 'RF')
visualize(y_test, rf_Pred, 'RF')
rf_auc = auc_roc_metrics(rf, X_test, y_test, 'RF')
metrics_results['rf'] = rf_auc


# Show the results of this model

# In[58]:


rf2 = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=None, max_features=None,
                       max_leaf_nodes=None, max_samples=0.8,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=10, min_samples_split=20,
                       min_weight_fraction_leaf=0.0, n_estimators=1000,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
rf2.fit(X_train_ds, y_train_ds)
rf2_Pred=rf2.predict(X_test)

#print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, rf2_Pred))
tn, fp, fn, tp = display_metrics(rf2, X_train_ds, X_test, y_train_ds, y_test, rf2_Pred, 'RF2')
visualize(y_test, rf2_Pred, 'RF2')
rf2_auc = auc_roc_metrics(rf2, X_test, y_test, 'RF2')
metrics_results['rf2'] = rf2_auc


# Try an unsupervised method using anamoly detection
# using PT gives a slightly better result

# In[59]:


rng = np.random.RandomState(42)
iso = IsolationForest(max_samples=100, random_state=rng, contamination=0.999, n_estimators=100, max_features=1.0)
iso.fit(X_train_pt)
iso_Pred = iso.predict(X_test_pt)
iso_Pred[iso_Pred == -1] = 0


# In[60]:


#print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, iso_Pred))
#tn, fp, fn, tp = display_metrics(iso, X_train, X_test, y_train, y_test, iso_Pred, 'ISO')
visualize(y_test, iso_Pred, 'ISO')
#iso_auc = auc_roc_metrics(iso, X_test, y_test, 'ISO')
#metrics_results['iso'] = iso_auc


# There is some variability in the results from run to run, due to random sampling and imbalanced data. This time, LogisticRegression has better prediction capability, the RandomForestClassifier test has a lot more mistakes in the False Positive category, and even a few more mistakes in the False Negative category.

# Now lets try a normal GradientBoosting Algorithm

# In[61]:


weight_for_0 = (1 / neg)*(total)/5.0 # was 5
weight_for_1 = (1 / pos)*(total)/2.0 # was 2

#setup model parameters, change some of the defaults based on benchmarking
gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=5, 
                                    max_depth=3, random_state=2, subsample = 0.5, criterion='mse', 
                                    min_samples_split = 10, min_samples_leaf = 10)

#since a false negative is much more likely than a false positive, we should weight them accordingly
gb_clf.fit( X_train_sc, y_train, sample_weight=np.where(y_train == 1, weight_for_1, weight_for_0).flatten()) #  fn = 12 and fp = 1057
# no weights gives worse false positive counts
#gb_clf.fit( X_train_ds, y_train_ds) # fn = 8 and fp = 2639

#use model to predict validation dataset
predictions = gb_clf.predict(X_test_sc)
tn, fp, fn, tp = display_metrics(gb_clf, X_train_sc, X_test_sc, y_train, y_test, predictions, 'GB')
visualize(y_test, predictions, 'GB')
gb_auc = auc_roc_metrics(gb_clf, X_test_sc, y_test, 'GB')
metrics_results['gb'] = gb_auc


# Display the results

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
# build the 1st model to be used later on the validate dataset - optimize on high NPV (Negative Predictive Value) metric

# Optimize model using GridSearchCV

# In[62]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer


# In[63]:


#creating Scoring parameter:

# need this because npv is not available for scoring, but precision is just npv with 0 and 1 swapped
inv_y_train = 1 - y_train

scoring = {
    'precision': make_scorer(precision_score),
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0)
    
}

# A sample parameter

parameters = {
    "loss":["deviance"],
    "learning_rate": [0.5],#, 0.01],
    "min_samples_split": [0.1],
    "min_samples_leaf": [0.05],
    "max_depth":[4],
    "max_features":["log2"],
    "criterion": ["mse"],
    "subsample":[0.5],#, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[20],
    "tol": [0.001],
    "ccp_alpha": [0.001],
    "random_state": [0]
    }
#passing the scoring function in the GridSearchCV
clf = GridSearchCV(GradientBoostingClassifier(), parameters,scoring=scoring,refit=False,cv=10, n_jobs=-1)

#clf.fit(X_train, y_train)
clf.fit(X_train, inv_y_train, sample_weight=np.where(inv_y_train == 0,3.6,1.4) ) # was 5.0
#converting the clf.cv_results to dataframe
df=pd.DataFrame.from_dict(clf.cv_results_)
df.to_csv('gb1.csv',index=False)
#here Possible inputs for cross validation is cv=2, there two split split0 and split1
df[['split0_test_precision','split1_test_precision','split0_test_accuracy','split1_test_accuracy','split0_test_sensitivity','split1_test_sensitivity','split0_test_specificity','split1_test_specificity']]


# In[64]:


df


# In[65]:


weight_for_0 = (1 / neg)*(total)/5.0 # was 5
weight_for_1 = (1 / pos)*(total)/2.0 # was 2

#setup model parameters, change some of the defaults based on benchmarking
gb_clf1 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=5, 
                                    max_depth=3, random_state=0, subsample = 0.5, criterion='mse', 
                                    min_samples_split = 10, min_samples_leaf = 10)

#since a false negative is much more likely than a false positive, we should weight them accordingly
gb_clf1.fit( X_train_sc, y_train, sample_weight=np.where(y_train == 1, weight_for_1, weight_for_0).flatten()) #  fn = 12 and fp = 1057
# no weights gives worse false positive counts
#gb_clf1.fit( X_train_ds, y_train_ds) # fn = 8 and fp = 2639

#use model to predict validation dataset
predictions = gb_clf1.predict(X_test_sc)
algo = 'GB1 Train **'
tn1, fp1, fn1, tp1 = display_metrics(gb_clf1, X_train_sc, X_test_sc, y_train, y_test, predictions, algo)
visualize(y_test, predictions, algo)
gb1_auc = auc_roc_metrics(gb_clf1, X_test_sc, y_test, algo)
metrics_results['gb1_train'] = gb1_auc


# 2nd step takes all the Predicted Positives (the misclassified FP from upper right (~ 14000) plus the TP (since we won't use the actual value until the validation step)) and reprocesses these using a different model. The other 2 squares (Predicted 0's) are not included in the 2nd model, since we already have a low False negative result, so the initial predicted 0s don't change. Will need to add those back into the final results at the end.

# Add 1st model prediction column to X_test for filtering

# In[66]:


X_test_sc['Prediction'] = predictions
# select rows with prediction of 1
yes_ind = X_test_sc[X_test_sc['Prediction'] == 1].index
# Create 2nd train dataset from 1st dataset where the prediction was 1
X2_test = X_test_sc.loc[yes_ind]
y2_test = y_test.loc[yes_ind]
y_test

# clean up the X_test dataset for future modeling, means remove the Prediction column
X_test_sc = X_test_sc.drop(['Prediction'], axis=1)
X2_test = X2_test.drop(['Prediction'], axis=1)


# Look at the prediction values from the first model (preda_1) for the rows with a predicted label of 0

# In[67]:


proba = gb_clf1.predict_proba(X2_test) 
pred = gb_clf1.predict(X2_test) 
df = pd.DataFrame(data=proba[:,0], columns=["preda_1"])
df.hist(bins=20, figsize=(10,5))
plt.show()


# Then we look at the ROC curve

# In[68]:


algo = 'PredictedPositives'
test_labels_temp = y2_test
ns_probs = [0 for _ in range(len(test_labels_temp))]
auc_roc_metrics_plots(proba[:,1], ns_probs, test_labels_temp, algo)


# Next we build the 2nd model to be used model later on the validate dataset and look at the output - optimize on Specificity metric  
# 2nd step: optimize to lowest False Positive (Type 1 error) counts = high specificity  tn/(tn+fp)

# In[69]:


#creating Scoring parameter:

scoring = {
    'specificity': make_scorer(recall_score,pos_label=0),
    'precision': make_scorer(precision_score),
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score)
}

# A sample parameter

parameters = {
    "loss":["deviance"],
    "learning_rate": [0.05],#, 0.01],
    "min_samples_split": [0.1],
    "min_samples_leaf": [0.1],
    "max_depth":[4],#,5,8],
    "max_features":["log2"],
    "criterion": ["mse"],
    "subsample":[0.5],#, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[20],
    "tol": [0.001],
    "ccp_alpha": [0.05] 
}

#passing the scoring function in the GridSearchCV
clf = GridSearchCV(GradientBoostingClassifier(), parameters,scoring=scoring,refit=False,cv=10, n_jobs=-1)

clf.fit(X_train_sc, y_train, sample_weight=np.where(y_train == 1, weight_for_1, weight_for_0).flatten())
#converting the clf.cv_results to dataframe
df=pd.DataFrame.from_dict(clf.cv_results_)
df.to_csv('gb1.csv',index=False)
#here Possible inputs for cross validation is cv=2, there two split split0 and split1
df[['split0_test_specificity','split1_test_specificity','split0_test_accuracy','split1_test_accuracy','split0_test_precision','split1_test_precision','split0_test_sensitivity','split1_test_sensitivity']]


# In[70]:


weight_for_0 = (1 / neg)*(total)/2.0 # was 5
weight_for_1 = (1 / pos)*(total)/5.0 # was 2

#setup model parameters, change some of the defaults based on benchmarking
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=5, 
                                    max_depth=3, random_state=0, subsample = 0.5, criterion='mse', 
                                    min_samples_split = 10, min_samples_leaf = 10)

#since a false negative is much more likely than a false positive, we should weight them accordingly
gb_clf2.fit( X_train_sc, y_train, sample_weight=np.where(y_train == 1, weight_for_1, weight_for_0).flatten()) #  fn = 12 and fp = 1057
# no weights gives worse false positive counts
#gb_clf1.fit( X_train_ds, y_train_ds) # fn = 8 and fp = 2639

#use model to predict validation dataset
predictions = gb_clf2.predict(X2_test) 

algo = 'GB2 Train **'
tn, fp, fn, tp = display_metrics(gb_clf2, X_train_sc, X2_test, y_train, y2_test, predictions, algo)

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

# In[71]:


# run the validate dataset through the first model
algo = '2-Step'
predictions1 = gb_clf1.predict(X_val_sc)
predictions_proba1 = gb_clf1.predict_proba(X_val_sc)
X1_val_final = X_val.copy()
X1_val_final=X1_val_final.join(y_val)
X1_val_final['Proba_1'] = predictions_proba1[:,1]
#X1_val_final
#X_val = X_val.sort_index(axis = 0) 


# In[72]:


# adding this
# use both models to predict final validation dataset
algo = 'GB1 Validate **'
tn1, fp1, fn1, tp1 = display_metrics(gb_clf1, X_test_sc, X_val_sc, y_test, y_val, predictions1, algo) 
visualize(y_val, predictions1, algo)
gb1_auc = auc_roc_metrics(gb_clf1, X_val_sc, y_val, algo)
metrics_results['gb1_validate'] = gb1_auc


# In[73]:



X_val_sc['Prediction'] = predictions1

yes_ind = X_val_sc[X_val_sc['Prediction'] == 1].index

X2_val = X_val_sc.loc[yes_ind]
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
X_val_sc = X_val_sc.drop(['Prediction'], axis=1)


# In[74]:


algo = 'GB2 Validate **'
tn, fp, fn, tp = display_metrics(gb_clf2, X_train_sc, X2_val, y_train, y2_val, predictions2, algo) 
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
print('SS Avg = ', (Specificity + Sensitivity)/2)

print('2 Step Algorithm' + ':TEST | AUC Score: ' + str( round( (Specificity+Sensitivity)/2,3 )))

cm_results.append([algo, (tn+tn1), fp, (fn+fn1), tp])
# HERE
#two_step_auc = auc_roc_metrics(gb_clf, X_test_sc, y_test, '2-Step')


# In[75]:


# try to combine the 2 models into one AUC score, however not sure that the proba values from 2 different models can be combined 

test_labels_temp = X_val_final['Class']
ns_probs = [0 for _ in range(len(test_labels_temp))]
model_probs = X_val_final['Proba_2']
model_pred=[1 if i > 0.50 else 0 for i in model_probs]

two_step_auc = auc_roc_metrics_plots(model_probs, ns_probs, test_labels_temp, algo)

metrics_results['2-step'] = two_step_auc

cr_results.append([algo, classification_report(test_labels_temp, model_pred)])


# In[76]:


y=np.reshape(test_labels_temp.to_numpy(), -1)
fpr, tpr, thresholds = metrics.roc_curve(y, model_probs, pos_label=1)
roc.append([algo, fpr, tpr, thresholds])


# 2nd round:

# In[77]:


with MyTimer(): 
    # 1st step: optimize to lowest False Negative (Type 2 error) counts = high npv  tn/(tn+fn)
    # 2nd step: optimize to lowest False Positive (Type 1 error) counts = high specificity  tn/(tn+fp)

    # can add: validation_fractionfloat=0.1, default=0.1
    #          n_iter_no_changeint=20, default=None

    neg, pos = np.bincount(train_labels_df['Class'])
    total = neg + pos
    initial_bias = np.log([pos/neg])

    weight_for_0 = (1 / neg)*(total)/5.0 # was 5
    weight_for_1 = (1 / pos)*(total)/2.0 # was 2
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('0:', weight_for_0, '1:', weight_for_1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=0, patience=25, mode='max', restore_best_weights=True)

    # can try Bayesian, could be better and boosting_type='Plain'
    gb_clf1 = CatBoostClassifier(iterations=200, learning_rate=0.01, class_weights=class_weight,
        eval_metric='Logloss', random_seed=15, custom_loss=['AUC'], loss_function='Logloss', # random_seed was 0
        bootstrap_type='MVS', subsample=0.76, mvs_reg=0.0, random_strength=1.0, use_best_model=True,       
        max_depth=6, boosting_type='Ordered',  boost_from_average=False
    )
    #gb_clf1.set_scale_and_bias(1.0, initial_bias)
    #since a false negative is much more likely than a false positive, we should weight them accordingly. 
    #IE Finding a true one is more important, also more rare
    gb_clf1.fit(train_features_df, train_labels_df, 
            #cat_features=cat_features, 
            eval_set=(val_features_df, val_labels_df), 
            verbose=False,
    )

    predictions = gb_clf1.predict(test_features_df) 

    print(metrics.confusion_matrix(test_labels_df, predictions))
    tn1, fp1, fn1, tp1 = confusion_matrix(test_labels_df, predictions).ravel()

    # Add 1st model's prediction as new column to X_test for filtering

    test_features_df['Prediction'] = predictions
    # select rows with prediction of 1
    yes_ind = test_features_df[test_features_df['Prediction'] == 1].index
    # Create 2nd train dataset from 1st dataset where the prediction was 1
    X2_test = test_features_df.loc[yes_ind]
    y2_test = test_labels_df.loc[yes_ind]

    # clean up the X_test dataset for future modeling, means remove the Prediction column
    test_features_df = test_features_df.drop(['Prediction'], axis=1)
    X2_test = X2_test.drop(['Prediction'], axis=1)

    # Next we build the 2nd model to be used model later on the validate dataset and look at the output - optimize on Specificity metric  
    # 2nd step: optimize to lowest False Positive (Type 1 error) counts = high specificity  tn/(tn+fp)

    second_model='dnn'
    if (second_model == 'dnn'):
        gb_clf2 = make_model(output_bias = initial_bias, opt_sel='sgd')

        weighted_history = gb_clf2.fit(
            train_features_df, train_labels_df, initial_epoch=0, batch_size=BATCH_SIZE,
            epochs=EPOCHS, callbacks = [early_stopping], #callbacks = callbacks_list,
            validation_data=(val_features_df, val_labels_df), shuffle=False, verbose=0, class_weight=class_weight
        ) 
        predictions = gb_clf2.predict_classes(X2_test, verbose=verbose, batch_size=BATCH_SIZE)
    elif (second_model=='gboost'):
        gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.01, max_features="log2", 
                                            max_depth=4, random_state=0, subsample = 0.5, criterion='mse',
                                            min_samples_split = 0.1, min_samples_leaf = 0.1, tol = 0.001, ccp_alpha = 0.05)

        # since a false negative is much more likely than a false positive, we should weight them accordingly. 
        # IE Finding a true one is more important
        # note that the weights in the 2nd model are the inverse of the weights in the 1st model
        gb_clf2.fit( X_train, y_train, sample_weight=np.where(y_train == 1, weight_for_1, weight_for_0).flatten()) # was 0.1 but should be > 1 to work correctly

        #use model to predict validation dataset
        predictions = gb_clf2.predict(X2_test)
    elif (second_model=='logreg'):
        gb_clf2 = LogisticRegression()
        #lr = LogisticRegression(solver='lbfgs')
        #gb_clf2.fit(X_train_ds, y_train_ds)
        
        gb_clf2.fit( X_train, y_train, sample_weight=np.where(y_train == 1, weight_for_1, weight_for_0).flatten()) # was 0.1 but should be > 1 to work correctly

        predictions = gb_clf2.predict(X2_test)
        #redictions = prediction_cutoff(gb_clf2, X2_test, 0.5) # 0.5 is the default cutoff for a logistic regression test
    else:
        print("no second model selected!!")

    print(metrics.confusion_matrix(y2_test, predictions))
    tn, fp, fn, tp = confusion_matrix(y2_test, predictions).ravel()
    print("combined test:", tn+tn1, fp, fn+fn1, tp)

    #************ run the validate dataset through the first model ****************
    algo = '2-Step'
    predictions1 = gb_clf1.predict(X_val_sc)
    tn1, fp1, fn1, tp1 = confusion_matrix(y_val, predictions1).ravel()
    print(metrics.confusion_matrix(y_val, predictions1))

    predictions_proba1 = gb_clf1.predict_proba(X_val_sc)
    X1_val_final = X_val_sc.copy()
    X1_val_final=X1_val_final.join(y_val)
    X1_val_final['Proba_1'] = predictions_proba1[:,1]

    # adding this
    # use both models to predict final validation dataset

    X_val_sc['Prediction'] = predictions1
    # add column for filtering
    yes_ind = X_val_sc[X_val_sc['Prediction'] == 1].index

    # filter dataset to include only the predicted value of 1
    X2_val = X_val_sc.loc[yes_ind]
    y2_val = y_val.loc[yes_ind]
    X2_val = X2_val.drop(['Prediction'], axis=1)
    # after filtering, run the validate dataset through the second model. Need to handle keras neural networks differently
    if ("keras" in str(gb_clf2)):
        predictions2 = gb_clf2.predict_classes(X2_val)
    else:
        predictions2 = gb_clf2.predict(X2_val)

    X2_val_final = X2_val.copy()
    X2_val_final.join(y2_val)
    
    predictions_proba2 = gb_clf2.predict_proba(X2_val)
    # validate the join!!
    X2_val_final['Proba_2'] = predictions_proba2[:, -1]

    cols_to_use = X2_val_final.columns.difference(X1_val_final.columns)
    X_val_final = X1_val_final.join(X2_val_final[cols_to_use], how='left', lsuffix='_1', rsuffix='_2')
    X_val_final.loc[X_val_final['Proba_2'].isnull(),'Proba_2'] = X_val_final['Proba_1']

    #remove this column for use later
    X_val_sc = X_val_sc.drop(['Prediction'], axis=1)

    print(metrics.confusion_matrix(y2_val, predictions2))
    tn, fp, fn, tp = confusion_matrix(y2_val, predictions2).ravel()
    print("combined val:", tn+tn1, fp, fn+fn1, tp)

    algo = '2-Step_2nd'
    Specificity = (tn+tn1)/(tn+tn1+fp)
    Sensitivity = tp/(tp+fn+fn1)

    print("Specificity =", Specificity)
    print("Sensitivity =", Sensitivity)
    print('SS Avg = ', (Specificity + Sensitivity)/2)

    cm_results.append([algo, (tn+tn1), fp, (fn+fn1), tp])

    # try to combine the 2 models into one AUC score, however not sure that the proba values from 2 models from different families/scales can be combined 

    test_labels_temp = X_val_final['Class']
    ns_probs = [0 for _ in range(len(test_labels_temp))]
    model_probs = X_val_final['Proba_2']
    model_pred=[1 if i > 0.50 else 0 for i in model_probs]

    two_step_auc = auc_roc_metrics_plots(model_probs, ns_probs, test_labels_temp, algo)
    metrics_results['2-step'] = two_step_auc
    cr_results.append([algo, classification_report(test_labels_temp, model_pred)])

    y=np.reshape(test_labels_temp.to_numpy(), -1)
    fpr, tpr, thresholds = metrics.roc_curve(y, model_probs, pos_label=1)
    roc.append([algo, fpr, tpr, thresholds])


# In[78]:


if ( 'test_features_df' in globals() ):
    print("local")
elif ( 'test_features_df' in locals() ):
    print("global")


# In[79]:


datasets = [X_train, y_train, X_test, y_test, X_val, y_val]
datasets_ds = [X_train_ds, y_train_ds, X_test, y_test, X_val, y_val]
datasets_sc = [X_train_sc, y_train, X_test_sc, y_test, X_val_sc, y_val]
datasets_df = [train_features_df, train_labels_df, test_features_df, test_labels_df, val_features_df, val_labels_df]


# In[142]:



# 1st step: optimize to lowest False Negative (Type 2 error) counts = high npv  tn/(tn+fn)
# 2nd step: optimize to lowest False Positive (Type 1 error) counts = high specificity  tn/(tn+fp)

# can add: validation_fractionfloat=0.1, default=0.1
#          n_iter_no_changeint=20, default=None
    
def RunTwoModels(first_datasets, second_datasets, eval_datasets, first_model, second_model='dnn'):
    first_train_features = first_datasets[0].copy(deep=True)
    first_train_labels   = first_datasets[1].copy(deep=True)
    first_test_features  = first_datasets[2].copy(deep=True)
    first_test_labels    = first_datasets[3].copy(deep=True)
    first_val_features   = first_datasets[4].copy(deep=True)
    first_val_labels     = first_datasets[5].copy(deep=True)
    second_train_features= second_datasets[0].copy(deep=True)
    second_train_labels  = second_datasets[1].copy(deep=True)
    second_test_features = second_datasets[2].copy(deep=True)
    second_test_labels   = second_datasets[3].copy(deep=True)
    second_val_features  = second_datasets[4].copy(deep=True)
    second_val_labels    = second_datasets[5].copy(deep=True)
    eval_train_features  = eval_datasets[0].copy(deep=True)
    eval_train_labels    = eval_datasets[1].copy
    eval_test_features   = eval_datasets[2].copy(deep=True)
    eval_test_labels     = eval_datasets[3].copy(deep=True)
    eval_val_features    = eval_datasets[4].copy(deep=True)
    eval_val_labels      = eval_datasets[5].copy(deep=True)

    neg, pos = np.bincount(first_train_labels['Class'])
    total = neg + pos
    initial_bias = np.log([pos/neg])

    weight_for_0 = (1 / neg)*(total)/5.0 # was 5
    weight_for_1 = (1 / pos)*(total)/2.0 # was 2
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('0:', weight_for_0, '1:', weight_for_1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=0, patience=25, mode='max', restore_best_weights=True)

    # can try Bayesian, could be better and boosting_type='Plain'
    a='''
    gb_clf1 = CatBoostClassifier(iterations=200, learning_rate=0.01, class_weights=class_weight,
        eval_metric='Logloss', random_seed=15, custom_loss=['AUC'], loss_function='Logloss', # random_seed was 0
        bootstrap_type='MVS', subsample=0.76, mvs_reg=0.0, random_strength=1.0, use_best_model=True,       
        max_depth=6, boosting_type='Ordered',  boost_from_average=False
    )'''
    #gb_clf1.set_scale_and_bias(1.0, initial_bias)
    #since a false negative is much more likely than a false positive, we should weight them accordingly. ""
    #IE Finding a true one is more important, also more rare
    first_model.fit(first_train_features, first_train_labels, 
            #cat_features=cat_features, 
            eval_set=(first_val_features, first_val_labels), 
            verbose=False,
    )

    predictions = first_model.predict(first_test_features) 

    print(metrics.confusion_matrix(first_test_labels, predictions))
    tn1, fp1, fn1, tp1 = confusion_matrix(first_test_labels, predictions).ravel()

    # Add 1st model's prediction as new column to X_test for filtering

    first_test_features['Prediction'] = predictions
    # select rows with prediction of 1
    yes_ind = first_test_features[first_test_features['Prediction'] == 1].index
    # Create 2nd train dataset from 1st dataset where the prediction was 1
    X2_test = first_test_features.loc[yes_ind]
    y2_test = first_test_labels.loc[yes_ind]

    # clean up the X_test dataset for future modeling, means remove the Prediction column
    first_test_features = first_test_features.drop(['Prediction'], axis=1)
    X2_test = X2_test.drop(['Prediction'], axis=1)

    # Next we build the 2nd model to be used model later on the validate dataset and look at the output - optimize on Specificity metric  
    # 2nd step: optimize to lowest False Positive (Type 1 error) counts = high specificity  tn/(tn+fp)

    #second_model='dnn'
    if (second_model == 'dnn'):
        print("running dnn model")
        gb_clf2 = make_model(output_bias = initial_bias, opt_sel='sgd')

        weighted_history = gb_clf2.fit(
            second_train_features, second_train_labels, initial_epoch=0, batch_size=BATCH_SIZE,
            epochs=EPOCHS, callbacks = [early_stopping], #callbacks = callbacks_list,
            validation_data=(second_val_features, second_val_labels), shuffle=False, verbose=0, class_weight=class_weight
        ) 
        predictions = gb_clf2.predict_classes(X2_test, verbose=verbose, batch_size=BATCH_SIZE)
    elif (second_model=='gboost'):
        gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.01, max_features="log2", 
                                            max_depth=4, random_state=0, subsample = 0.5, criterion='mse',
                                            min_samples_split = 0.1, min_samples_leaf = 0.1, tol = 0.001, ccp_alpha = 0.05)

        # since a false negative is much more likely than a false positive, we should weight them accordingly. 
        # IE Finding a true one is more important
        # note that the weights in the 2nd model are the inverse of the weights in the 1st model
        gb_clf2.fit( second_train_features, second_train_labels, sample_weight=np.where(second_train_labels == 1, weight_for_1, weight_for_0).flatten()) # was 0.1 but should be > 1 to work correctly

        #use model to predict validation dataset
        predictions = gb_clf2.predict(X2_test)
    elif (second_model=='logreg'):
        gb_clf2 = LogisticRegression()
        #lr = LogisticRegression(solver='lbfgs')
        #gb_clf2.fit(X_train_ds, y_train_ds)
        
        gb_clf2.fit( second_train_features, second_train_labels, sample_weight=np.where(second_train_labels == 1, weight_for_1, weight_for_0).flatten()) # was 0.1 but should be > 1 to work correctly

        predictions = gb_clf2.predict(X2_test)
        #redictions = prediction_cutoff(gb_clf2, X2_test, 0.5) # 0.5 is the default cutoff for a logistic regression test
    else:
        print("no second model selected!!")

    print(metrics.confusion_matrix(y2_test, predictions))
    tn, fp, fn, tp = confusion_matrix(y2_test, predictions).ravel()
    print("combined test:", tn+tn1, fp, fn+fn1, tp)

    #************ run the validate dataset through the first model ****************
    algo = '2-Step'
    predictions1 = first_model.predict(eval_test_features)
    tn1, fp1, fn1, tp1 = confusion_matrix(eval_test_labels, predictions1).ravel()
    print(metrics.confusion_matrix(eval_test_labels, predictions1))

    predictions_proba1 = first_model.predict_proba(eval_test_features)
    X1_val_final = eval_test_features.copy()
    X1_val_final=X1_val_final.join(eval_test_labels)
    X1_val_final['Proba_1'] = predictions_proba1[:,1]

    # adding this
    # use both models to predict final validation dataset

    eval_test_features['Prediction'] = predictions1
    # add column for filtering
    yes_ind = eval_test_features[eval_test_features['Prediction'] == 1].index

    # filter dataset to include only the predicted value of 1
    X2_val = eval_test_features.loc[yes_ind]
    y2_val = eval_test_labels.loc[yes_ind]
    X2_val = X2_val.drop(['Prediction'], axis=1)
    # after filtering, run the validate dataset through the second model. Need to handle keras neural networks differently
    if ("keras" in str(gb_clf2)):
        predictions2 = gb_clf2.predict_classes(X2_val)
    else:
        predictions2 = gb_clf2.predict(X2_val)

    X2_val_final = X2_val.copy()
    X2_val_final.join(y2_val)
    
    predictions_proba2 = gb_clf2.predict_proba(X2_val)
    # validate the join!!
    X2_val_final['Proba_2'] = predictions_proba2[:, -1]

    cols_to_use = X2_val_final.columns.difference(X1_val_final.columns)
    X_val_final = X1_val_final.join(X2_val_final[cols_to_use], how='left', lsuffix='_1', rsuffix='_2')
    X_val_final.loc[X_val_final['Proba_2'].isnull(),'Proba_2'] = X_val_final['Proba_1']

    #remove this column for use later
    eval_test_features = eval_test_features.drop(['Prediction'], axis=1)

    print(metrics.confusion_matrix(y2_val, predictions2))
    tn, fp, fn, tp = confusion_matrix(y2_val, predictions2).ravel()
    print("combined val:", tn+tn1, fp, fn+fn1, tp)

    algo = '2-Step_func'
    Specificity = (tn+tn1)/(tn+tn1+fp)
    Sensitivity = tp/(tp+fn+fn1)

    print("Specificity =", Specificity)
    print("Sensitivity =", Sensitivity)
    print('SS Avg = ', (Specificity + Sensitivity)/2)

    cm_results.append([algo, (tn+tn1), fp, (fn+fn1), tp])

    # try to combine the 2 models into one AUC score, however not sure that the proba values from 2 models from different families/scales can be combined 

    test_labels_temp = X_val_final['Class']
    ns_probs = [0 for _ in range(len(test_labels_temp))]
    model_probs = X_val_final['Proba_2']
    model_pred=[1 if i > 0.50 else 0 for i in model_probs]

    two_step_auc = auc_roc_metrics_plots(model_probs, ns_probs, test_labels_temp, algo)
    metrics_results['2-step'] = two_step_auc
    cr_results.append([algo, classification_report(test_labels_temp, model_pred)])

    y=np.reshape(test_labels_temp.to_numpy(), -1)
    fpr, tpr, thresholds = metrics.roc_curve(y, model_probs, pos_label=1)
    roc.append([algo, fpr, tpr, thresholds])
    
    return(tn+tn1, fp, fn+fn1, tp)

with MyTimer():
    model1 = CatBoostClassifier(iterations=200, learning_rate=0.01, class_weights=class_weight,
        eval_metric='Logloss', random_seed=15, custom_loss=['AUC'], loss_function='Logloss', # random_seed was 0
        bootstrap_type='MVS', subsample=0.76, mvs_reg=0.0, random_strength=1.0, use_best_model=True,       
        max_depth=6, boosting_type='Ordered',  boost_from_average=False
    )
    model2 = LogisticRegression()
    (tn, fp, fn, tp) = RunTwoModels(datasets_df, datasets_df, datasets_sc, first_model = model1, second_model='dnn')


# The 2 step process has the highest sensitivity (and specificity) between the models. The 2 step process also improves the overall model prediction of positives by a large amount (FP/TP ratio from above 10x to below 2x). I don't think we could get this high of precision and recall together with a single model. The best I could do with a single model was 10x FP/TP ratio.

# Next will try a few Neural Networks

# Adding swish activation function code for possible use later, can compare to relu, etc

# In[91]:


# create new activation function
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))


# In[92]:


# add this function to the list of Activation functions
get_custom_objects().update({'swish': Activation(swish)})


# Create the models to be used layer, using Sequential()

# In[93]:


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


# In[94]:


def create_simple_dnn(input_dim):
    # input_dim must equal number of features in X_train and X_test dataset
    clf1 = Sequential([
        Dense(units=16, kernel_initializer='uniform', input_dim=input_dim, activation='relu'),
        Dropout(0.25),
        Dense(units=18, kernel_initializer='uniform', activation='relu'),
        Dense(1, kernel_initializer='uniform', activation='sigmoid')
    ])
    return clf1


# In[95]:


def create_dnn_complex(input_dim):
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


# In[96]:


# source: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
def create_online_dnn(input_dim, output_bias=0):
    output_bias = keras.initializers.Constant(output_bias)
    # input_dim must equal number of features in X_train and X_test dataset
    clf1 = Sequential([
        Dense(units=16, kernel_initializer='uniform', input_dim=input_dim, activation='relu'),
        Dropout(0.25),
        Dense(units=18, kernel_initializer='uniform', activation='relu'),
        Dense(1, kernel_initializer='uniform', activation='sigmoid', bias_initializer=output_bias)
        #Dense(1, activation='sigmoid', bias_initializer=output_bias)
    ])
    return clf1


# In[97]:


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


# In[98]:


class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.npv = []
        self.recall = []
        self.specificity = []
        self.f1s = []
        self.kappa = []
        self.auc = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        self.auc.append(sklm.roc_auc_score(targ, score))
        self.confusion.append(sklm.confusion_matrix(targ, predict))
        self.precision.append(sklm.precision_score(targ, predict))
        self.npv.append(sklm.precision_score(1-targ, 1-predict))
        self.recall.append(sklm.recall_score(targ, predict))
        self.specificity.append(sklm.recall_score(1-targ, 1-predict))
        self.f1s.append(sklm.f1_score(targ, predict))
        self.kappa.append(sklm.cohen_kappa_score(targ, predict))
        print('precision: ', sklm.precision_score(targ, predict))
        print('recall: ', sklm.recall_score(targ, predict))
        print('npv: ', sklm.precision_score(1-targ, 1-predict))
        print('specificity: ', sklm.recall_score(1-targ, 1-predict))

        return


# In[99]:


import keras.backend as K
def custom_loss_abs_sum(y_true, y_pred):
    loss = abs(y_true - y_pred)
    return loss
#clf.compile(optimizer='adam', loss=custom_loss_abs_sum, metrics=['accuracy']) 


# run the CNN model

# In[100]:


input_shape = (X_train_ds.shape[1], 1)
input_dim = X_train_ds.shape[1]
print("Input shape:", input_shape)
clf = create_cnn(input_shape)
# NameError: name 'input_shape' is not defined

# reshape data for CNN expected input
nrows, ncols = X_train_ds.shape # (602,30)
X_train_arr = X_train_ds.copy().to_numpy()
y_train_arr = y_train_ds.copy()
X_train_arr = X_train_arr.reshape(nrows, ncols, 1)

nrows, ncols = X_test.shape # (602,30)
X_test_arr = X_test.copy().to_numpy()
y_test_arr = y_test.copy()
X_test_arr = X_test_arr.reshape(nrows, ncols, 1)

nrows, ncols = X_val.shape # (602,30)
X_val_arr = X_val.copy().to_numpy()
y_val_arr = y_val.copy()
X_val_arr = X_val_arr.reshape(nrows, ncols, 1)

#opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
# Let's train the model using RMSprop
#clf.compile(loss='binary_crossentropy',
#              optimizer=opt,
#              metrics=['accuracy'])
# or
#clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
clf.compile(optimizer='adam', loss=custom_loss_abs_sum, metrics=['accuracy'])  

clf.summary()

#adam = keras.optimizers.Adam(learning_rate=0.001)
# try using focal_loss to give heavier weight to examples that are difficult to classify
# seems to improve the metrics slightly
#clf.compile(optimizer=adam, loss=[focal_loss], metrics=['accuracy'])

# create/fit model on the training dataset
#clf.fit(X_train, y_train, batch_size=16, epochs=32, sample_weight=np.where(y_train == 1,0.2,1.0).flatten())
#clf.fit(X_train, y_train, batch_size=16, epochs=20, sample_weight=np.where(y_train == 1,1.0,1.0).flatten())
# or

clf.fit(X_train_arr, y_train_arr, epochs=20, verbose=1, 
        #sample_weight=np.where(y_train_ds == 1, weight_for_1, weight_for_0).flatten(), 
        shuffle=True, validation_data=(X_val_arr, y_val_arr))
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
print("Classification Report (CNN)") 
print(classification_report(y_test_arr, yhat_classes))

tn, fp, fn, tp = display_metrics(clf, X_train_arr, X_test_arr, y_train_arr, y_test_arr, yhat_classes, 'CNN')
visualize(y_test_arr, yhat_classes, 'CNN')
cnn_auc = auc_roc_metrics(clf, X_test_arr, y_test_arr, 'CNN')
metrics_results['cnn'] = cnn_auc


# Now run the basic DNN (Deep Neural Network)

# for Custom Loss fit to Train dataset (balanced):  
# 
# Use Keras/Tensor math functions!  
# reference: https://www.tensorflow.org/api_docs/python/tf/keras/backend/  
# 
# Predictions on test dataset:  
# 
# loss = y_true - y_pred => all predictions are 1, fn are loss=+1, fp are loss=-1  
# [[    0 59708]  
# [    0   102]]  
# 
# loss = y_pred - y_true => all predictions are 0, fp are loss=+1, fn are loss=-1  
# [[59708     0]  
#  [  102     0]]  
#  
# loss = abs(y_true - y_pred) => all mistakes have equal weight  
# 
# [[57693  2015]  
#  [   12    90]]  

# In[103]:


# https://www.tensorflow.org/api_docs/python/tf/keras/backend/

import keras.backend as K
def custom_loss_mask(y_true, y_pred):
    #loss = abs(y_true - y_pred)
    
    mask1 = K.less(y_pred, y_true) # is y_pred < y_true or y_pred - y_true < 0, FN
    mask2 = K.less(y_true, y_pred) # is y_true < y_pred or y_true - y_pred < 0, FP
    #loss = K.cast(mask1, K.floatx()) * 2 * (y_true - y_pred) # only include FN
    loss = (K.cast(mask1, K.floatx()) * 2 * (y_true - y_pred)) + (K.cast(mask2, K.floatx()) * 4 * (y_pred - y_true)) # only include FN
    return loss
#clf.compile(optimizer='adam', loss=custom_loss_mask, metrics=['accuracy']) 


# In[104]:


# define variable learning rate function
from keras.callbacks import LearningRateScheduler, EarlyStopping, History, LambdaCallback
import math

def step_decay(epoch, lr):
    drop = 0.995 # was .999
    epochs_drop = 5.0 # was 175, sgd likes 200+, adam likes 100
    lrate = lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print("epoch=" + str(epoch) + " lr=" + str(lr) + " lrate=" + str(lrate))
    return lrate
lrate = LearningRateScheduler(step_decay)
early_stopping = EarlyStopping(monitor='val_loss', patience=25, mode='auto', restore_best_weights = True)
callbacks_list = [lrate, early_stopping] 


# In[105]:


with MyTimer(): 
    from keras.optimizers import Adam, SGD, RMSprop

    verbose=1
    clf = create_dnn(input_dim)
    clf.summary()
    #clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    learning_rate = 0.0001
    decay = 0.0002
    momentum=0.99
    opt_sel = "adam"
    if (opt_sel == "adam"):
        #opt = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad) # added to v86
        opt = Adam(lr=learning_rate)
    elif(opt_sel == "sgd"):
        opt = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay, nesterov=True)

    clf.compile(optimizer=opt, loss=custom_loss_mask, metrics=['accuracy'])

    #adam = keras.optimizers.Adam(learning_rate=0.001)
    # try using focal_loss to give heavier weight to examples that are difficult to classify
    # seems to improve the metrics slightly
    #clf.compile(optimizer=adam, loss=[focal_loss], metrics=['accuracy'])

    # create/fit model on the training dataset
    #clf.fit(X_train, y_train, batch_size=16, epochs=32, sample_weight=np.where(y_train == 1,0.2,1.0).flatten())
    clf.fit(X_train_ds, y_train_ds, batch_size=16, epochs=200, verbose=verbose, 
            #sample_weight=np.where(y_train == 1, weight_for_1, weight_for_0).flatten(),
            callbacks=callbacks_list, validation_data=(X_val, y_val))

    # check model metrics
    score = clf.evaluate(X_train_ds, y_train_ds, batch_size=128)
    print('\nAnd the Train Score is ', score[1] * 100, '%')
    score = clf.evaluate(X_test, y_test, batch_size=128)
    print('\nAnd the Test Score is ', score[1] * 100, '%')
    # predict probabilities for test set
    yhat_probs = clf.predict(X_test, verbose=verbose)
    # predict crisp classes for test set
    yhat_classes = clf.predict_classes(X_test, verbose=verbose)
    print("Classification Report (DNN)") 
    print(classification_report(y_test, yhat_classes))

    tn, fp, fn, tp = display_metrics(clf, X_train_ds, X_test, y_train_ds, y_test, yhat_classes, 'DNN')
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

# In[107]:


with MyTimer(): 
    clf = create_simple_dnn(input_dim)
    clf.summary()
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # create/fit model on the training dataset
    #clf.fit(X_train, y_train, batch_size=15, epochs=5, sample_weight=np.where(y_train == 1,0.1,1.0).flatten())
    clf.fit(X_train, y_train, batch_size=4096, epochs=128, verbose=verbose, sample_weight=np.where(y_train == 1, weight_for_1, weight_for_0).flatten(), 
            callbacks=callbacks_list, validation_data=(X_val, y_val))
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
    print("Classification Report (DNN Simple)") 
    print(classification_report(y_test, yhat_classes))
    tn, fp, fn, tp = display_metrics(clf, X_train, X_test, y_train, y_test, yhat_classes, 'DNN Simple')
    visualize(y_test, yhat_classes, 'DNN Simple')
    dnn_simple_auc = auc_roc_metrics(clf, X_test, y_test, 'DNN-Simple')
    metrics_results['dnn_simple'] = dnn_simple_auc


# In[108]:


def plot_metrics(history):
    metrics =  ['loss', 'auc', 'precision', 'recall', 'accuracy', 'crossentropy','fn','fp']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,4,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        elif metric == 'fn':
            plt.ylim([0,100])
        elif metric == 'fp':
            plt.ylim([0,5000])
        elif metric == 'accuracy':
            plt.ylim([0.8,1])
        elif metric == 'crossentropy':
            plt.ylim([0,0.2])
        elif metric == 'precision':
            plt.ylim([0,0.2])
        elif metric == 'recall':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()


# ![image.png](attachment:b642217d-f6e1-4c32-aaaa-345d247f3dce.png)

# In[110]:


with MyTimer(): 
    #early_stopping = EarlyStopping(monitor='val_loss', patience=25, mode='auto', restore_best_weights = True)
    #callbacks_list = [lrate, early_stopping] 
    neg, pos = np.bincount(y_train)
    initial_bias = np.log([pos/neg])
    clf = create_online_dnn(input_dim=input_dim, output_bias = initial_bias)
    clf.summary()
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
    # create/fit model on the training dataset
    baseline_history = clf.fit(X_train_sc, y_train, batch_size=32, epochs=100, 
            verbose=verbose, sample_weight=np.where(y_train == 1, weight_for_1, weight_for_0).flatten(),
            callbacks=callbacks_list, validation_data=(X_val_sc, y_val))
    # check model metrics
    score = clf.evaluate(X_train_sc, y_train, batch_size=128)
    print('\nAnd the Train Score is ', score[1] * 100, '%')
    print('\nThe loss is ',score[0])
    score = clf.evaluate(X_test_sc, y_test, batch_size=128)
    print('\nAnd the Test Score is ', score[1] * 100, '%')
    # predict probabilities for test set
    yhat_probs = clf.predict(X_test_sc, verbose=verbose)
    # predict crisp classes for test set
    yhat_classes = clf.predict_classes(X_test_sc, verbose=verbose)
    print("Classification Report (DNN Online)") 
    print(classification_report(y_test, yhat_classes))
    tn, fp, fn, tp = display_metrics(clf, X_train_sc, X_test_sc, y_train, y_test, yhat_classes, 'DNN Online')
    visualize(y_test, yhat_classes, 'DNN Online')
    dnn_online_auc = auc_roc_metrics(clf, X_test_sc, y_test, 'DNN-Online')
    metrics_results['dnn_online'] = dnn_online_auc
    mpl.rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #plot_metrics(baseline_history)


# This DNN is successful at reducing the FP/TP ratio. This is expected as a Neural Network can decide on its own rules to include based on the input data. Below I try other more and less complex methods, but so far the results are not as good.

# In[111]:


with MyTimer(): 
    clf = create_dnn_complex(input_dim)
    clf.summary()
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # create/fit model on the training dataset
    #clf.fit(X_train, y_train, batch_size=15, epochs=5, sample_weight=np.where(y_train == 1,0.1,1.0).flatten())
    clf.fit(X_train, y_train, batch_size=2048, epochs=128, 
           verbose=verbose, sample_weight=np.where(y_train == 1, weight_for_1, weight_for_0).flatten(),
           #callbacks=[early_stopping],
           validation_data=(X_val, y_val))
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
    print("Classification Report (DNN complex)") 
    print(classification_report(y_test, yhat_classes))
    tn, fp, fn, tp = display_metrics(clf, X_train, X_test, y_train, y_test, yhat_classes, 'DNN Complex')
    visualize(y_test, yhat_classes, 'DNN Complex')
    dnn_complex_auc = auc_roc_metrics(clf, X_test, y_test, 'DNN-Complex')
    metrics_results['dnn_complex'] = dnn_complex_auc


# In[112]:


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)


# In[113]:


neg, pos = np.bincount(train_labels)
total = neg + pos
initial_bias = np.log([pos/neg])

weight_for_0 = (1 / neg)*(total)/5.0 # was 5
weight_for_1 = (1 / pos)*(total)/2.0 # was 2

#weight_for_0 = 1
#weight_for_1 = neg/pos


class_weight = {0: weight_for_0, 1: weight_for_1}
print('0:', weight_for_0, '1:', weight_for_1)
initial_bias = 1.0 * initial_bias
print('initial_bias:', initial_bias)

clf = CatBoostClassifier(
    iterations=200, 
    learning_rate=0.01, 
    class_weights=class_weight,
    eval_metric='Logloss', # Logloss is default
    random_seed=0,
    custom_loss=['AUC'],
    loss_function='Logloss',
    bootstrap_type='MVS', # can try Bayesian, could be better
    subsample=0.76,
    mvs_reg=0.0,
    random_strength=1.0,
    use_best_model=True,
    max_depth=6,
    boosting_type='Ordered', # Plain
    boost_from_average=False,
    #l2_leaf_reg=0.1, # any reg decreases score
    #loss_function='CrossEntropy'
)
clf.set_scale_and_bias(1.0, initial_bias)

clf.fit(train_features, train_labels, 
        #cat_features=cat_features, 
        eval_set=(val_features, val_labels), 
        verbose=False,
        plot=True,
)

print('CatBoost model is fitted: ' + str(clf.is_fitted()))
print('CatBoost model parameters:')
print(clf.get_params())
yhat_classes = clf.predict(test_features, verbose=verbose, prediction_type='Class')
print(metrics.confusion_matrix(test_labels, yhat_classes))
tn, fp, fn, tp = display_metrics(clf, train_features, test_features, train_labels, test_labels, yhat_classes, 'CBoost_O')
dnn_weighted_auc = auc_roc_metrics(clf, test_features, test_labels, 'CBoost_O')
metrics_results['cb_ordered'] = dnn_weighted_auc


# In[114]:


X_test_sc.describe()


# In[115]:


test_features[1].mean()


# In[116]:


neg, pos = np.bincount(train_labels)
total = neg + pos
initial_bias = np.log([pos/neg])

weight_for_0 = (1 / neg)*(total)/5.0 
weight_for_1 = (1 / pos)*(total)/2.0

#weight_for_0 = 1
#weight_for_1 = neg/pos


class_weight = {0: weight_for_0, 1: weight_for_1}
print('0:', weight_for_0, '1:', weight_for_1)
initial_bias = 1.0 * initial_bias
print('initial_bias:', initial_bias)

clf = CatBoostClassifier(
    iterations=200, 
    learning_rate=0.01, 
    class_weights=class_weight,
    eval_metric='Logloss', # Logloss is default
    random_seed=0,
    custom_loss=['AUC'],
    loss_function='Logloss',
    bootstrap_type='MVS', # can try Bayesian, could be better
    subsample=0.76,
    mvs_reg=0.0,
    random_strength=1.0,
    use_best_model=True,
    max_depth=6,
    boosting_type='Plain', # Plain
    boost_from_average=False,
    #l2_leaf_reg=0.1, # any reg decreases score
    #loss_function='CrossEntropy'
)
#clf.set_scale_and_bias(1.0, initial_bias)

clf.fit(train_features, train_labels, 
        #cat_features=cat_features, 
        eval_set=(val_features, val_labels), 
        verbose=False,
        plot=True,
)

print('CatBoost model is fitted: ' + str(clf.is_fitted()))
print('CatBoost model parameters:')
print(clf.get_params())
yhat_classes = clf.predict(test_features, verbose=verbose, prediction_type='Class')
print(metrics.confusion_matrix(test_labels, yhat_classes))
tn, fp, fn, tp = display_metrics(clf, train_features, test_features, train_labels, test_labels, yhat_classes, 'CBoost_P')
dnn_weighted_auc = auc_roc_metrics(clf, test_features, test_labels, 'CBoost_P')
metrics_results['cb_plain'] = dnn_weighted_auc


# In[117]:


# use model from imbalanced_data notebook

BATCH_SIZE = 2048
EPOCHS = 100
neg, pos = np.bincount(train_labels)
total = neg + pos
initial_bias = np.log([pos/neg])
weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0
class_weight = {0: weight_for_0, 1: weight_for_1}
print('initial_bias:', initial_bias)
print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1));


# In[118]:


# for dnn model with weights and initial bias
def RunModel(mon='val_auc', mod='max', opt_sel='adam'):
    #First Pass
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=mon, verbose=0, patience=25, mode=mod, restore_best_weights=True)

    clf = make_model(output_bias = initial_bias, opt_sel=opt_sel)
    
    clf.fit(
        train_features,
        train_labels,
        initial_epoch=0,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks = [early_stopping],
        #callbacks = callbacks_list,
        validation_data=(val_features, val_labels),
        shuffle=False,
        verbose=0,
        # The class weights go here
        class_weight=class_weight
    ) 
    
    #clf.fit(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)#, shuffle=False)
    
    #evaluate does not change or train the model, think this can be skipped unless we want the loss metric
    results = clf.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
    print("Loss: {:0.4f}".format(results[0]))

    # 12/18 before removing these 4 lines
    initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')
    clf.save_weights(initial_weights)

    #clf = make_model()
    clf = make_model(output_bias = initial_bias, opt_sel=opt_sel)
    clf.load_weights(initial_weights)

    weighted_history = clf.fit(
        train_features,
        train_labels,
        initial_epoch=0,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks = [early_stopping],
        #callbacks = callbacks_list,
        validation_data=(val_features, val_labels),
        shuffle=False,
        verbose=0,
        # The class weights go here
        class_weight=class_weight
    ) 

    yhat_classes = clf.predict_classes(test_features, verbose=verbose, batch_size=BATCH_SIZE)
    print("mon='", mon, "',mod='", mod)
    print(metrics.confusion_matrix(test_labels, yhat_classes))
    tn, fp, fn, tp = display_metrics(clf, train_features, test_features, train_labels, test_labels, yhat_classes, 'DNN Wt '+opt_sel)
    dnn_weighted_auc = auc_roc_metrics(clf, test_features, test_labels, 'DNN Wt '+opt_sel)
    metrics_results['dnn_wt_'+opt_sel] = dnn_weighted_auc
    return (weighted_history);


# In[119]:


with MyTimer():
    #weighted_history = RunModel(mon='val_loss', mod='min');
    weighted_history = RunModel(mon='val_auc', mod='max', opt_sel='adam');


# In[120]:


with MyTimer():
    #weighted_history = RunModel(mon='val_loss', mod='min');
    weighted_history = RunModel(mon='val_auc', mod='max', opt_sel='sgd');


# In[121]:


plot_metrics(weighted_history)


# In[122]:


cm_results


# In[123]:


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


# In[124]:


# show all variables in memory
#%who or %whos


# In[125]:


clf = create_autoencoder(input_dim)
clf.summary()
#clf.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# create/fit model on the training dataset
#clf.fit(X_train, y_train, batch_size=32, epochs=32, shuffle=True)#, validation_data=(X_test, X_test))
clf.fit(X_train_ds, y_train_ds, batch_size=16, epochs=32, verbose=verbose, #sample_weight=np.where(y_train == 1, weight_for_1, weight_for_0).flatten(),
        callbacks=callbacks_list, validation_data=(X_val, y_val))
#clf.fit(X_train, y_train, batch_size=32, epochs=32, sample_weight=np.where(y_train == 1,0.1,1.0).flatten())
#clf.fit(X_train, y_train, batch_size=15, epochs=5)

# check model metrics
score = clf.evaluate(X_train_ds, y_train_ds, batch_size=32)
print('\nAnd the Train Score is ', score[1] * 100, '%')
score = clf.evaluate(X_test, y_test, batch_size=32)
print('\nAnd the Test Score is ', score[1] * 100, '%')
# predict probabilities for test set
yhat_probs = clf.predict(X_test, verbose=verbose)
# predict crisp classes for test set
yhat_classes = clf.predict_classes(X_test, verbose=verbose)
print("Classification Report (AutoEncoder)") 
print(classification_report(y_test, yhat_classes))
tn, fp, fn, tp = display_metrics(clf, X_train_ds, X_test, y_train_ds, y_test, yhat_classes, 'AutoEncoder')
visualize(y_test, yhat_classes, 'AutoEncoder')
autoencoder_auc = auc_roc_metrics(clf, X_test, y_test, 'AutoEncoder')
metrics_results['autoencoder'] = autoencoder_auc


# In[126]:


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

# In[127]:


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

# In[128]:


y_val['Class'].value_counts()


# number of Actual 0 and 1 in the final test dataset for all other models
# "1" total should match the FN + TP

# In[129]:


y_test['Class'].value_counts()


# Here are the final results in tabular form. 

# In[130]:


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
filtered = final_results[~final_results.algo.str.contains('\*', regex= True, na=False)]
sort = filtered.sort_values(filtered.columns[7], ascending = False)
print(sort)
sort.to_csv('c:\\DataScience\\Repo\\Imbalanced_data\\CreditCardFraud\\working\\results.csv', sep=',', mode='a', encoding='utf-8', header=True)


# In[131]:


print('mean(Avg):', filtered['Avg'].mean())


# In[132]:


f = open('c:\\DataScience\\Repo\\Imbalanced_data\\CreditCardFraud\\working\\averages.txt', 'a+')
f.write(str(filtered['Avg'].mean()))
f.write("\n")
f.close()


# In[133]:


print("Start: ", StartTime)
print("End: ", datetime.datetime.now())


# things to try, calculate optimal weights
# 
# ![image.png](attachment:81cfe9b6-b296-4e68-bd5d-ba1b50f7a895.png)
# 
# reference: https://medium.com/rv-data/how-to-do-cost-sensitive-learning-61848bf4f5e7

# v4 - optimize GB for 2-Step
# 
# use other models for 2-Step model, instead of GB. LR first

# In[134]:


# loading model that was saved in another script
#clf = keras.models.load_model("sampled_model")


# In[135]:


get_ipython().run_cell_magic('javascript', '', 'Jupyter.notebook.session.delete();')


# In[ ]:




