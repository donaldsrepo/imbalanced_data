# Import Libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas_profiling as pp
import seaborn as sns
import os

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
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import pyplot
import zipfile

import tensorflow as tf
import random
debug = 0

def draw_contour(df2, x,y,z): # ex: draw_contour(df_sum_in, x_label, y_label, z1_label)
    if (debug == 1):
        print('draw_contour')
        print(df2, x,y,z)
    # setup axes for contour mapping which requires a meshgrid for x,y
    Z = df2.pivot_table(index=x, columns=y, values=z).T.values
    X_unique = np.sort( np.unique(df2[x]) )
    Y_unique = np.sort( np.unique(df2[y]) )
    X, Y = np.meshgrid(X_unique, Y_unique)
    pd.DataFrame(Z).round(3)
    from IPython.display import set_matplotlib_formats
    #%matplotlib inline
    set_matplotlib_formats('svg')
    # Initialize plot objects
    params = {"text.color" : "blue",
              "xtick.color" : "green",
              "ytick.color" : "green",
              "figure.figsize" : "5, 5"}
    plt.rcParams.update(params)
    # draw the contour using matplotlib.pyplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Generate a contour plot
    cp = ax.contour(X, Y, Z)
    ax.clabel(cp, fontsize=10)
    ax.set_xlabel(x, color='red')
    _ = ax.set_ylabel(y, color='red')
    ax.set_title(z, color='red')

    xs=df2[x]
    ys=df2[y]
    zs=df2[z]
    
    plt.scatter(xs,ys,c=zs)
    name = 'plot' + z + '.png'
    fig.savefig(name)
    
# Define our custom loss function
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def prediction_cutoff(model, test_features, cutoff):
    model.predict_proba(test_features)
    # to get the probability in each class, 
    # for example, first column [:,0] is probability of y=0 and second column [:,1] is probability of y=1.
    prob1 = model.predict_proba(test_features)[:,1] 
    predicted = [1 if i > cutoff else 0 for i in prob1]
    return predicted

def create_models(train_features, train_labels, test_features, test_labels, val_features, val_labels, algo, rn1, rn2, spl1, spl2, wt1_1, wt1_0, wt2_1, wt2_0):
    #setup model parameters, change some of the defaults based on benchmarking
    # parameter tuning: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    gb_clf1 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=5, 
                                        max_depth=3, random_state = None, subsample = 1.0, criterion='mse', 
                                        min_samples_split = 10, min_samples_leaf = 10) # random_state=0, 

    #since a false negative is much more likely than a false positive, we should weight them accordingly. 
    #IE Finding a true one is more important, also more rare
    if (debug == 1):
        print('create M1 train_labels:',train_labels)
        print('create M1: train_labels\n', train_labels['Class'].value_counts())
    gb_clf1.fit( train_features, train_labels, sample_weight=np.where(train_labels == 1, wt1_1, wt1_0) ) # was 5.0

    #use model to predict validation dataset
    predictions = gb_clf1.predict(test_features) 
    
    # 2nd step takes all the Predicted Positives (the misclassified FP from upper right (~ 14000) plus the TP (since we won't use the actual value until the validation step)) and reprocesses these using a different model. The other 2 squares (Predicted 0's) are not included in the 2nd model, since we already have a low False negative result, so the initial predicted 0s don't change. Will need to add those back into the final results at the end.

    # Add 1st model prediction column to test_features for filtering

    test_features_temp = test_features.copy()
    test_features_temp['Prediction'] = predictions

    # select rows with prediction of 1

    yes_ind = test_features_temp[test_features_temp['Prediction'] == 1].index

    # Create 2nd train dataset from 1st dataset where the prediction was 1

    X2_test = test_features_temp.loc[yes_ind]
    y2_test = test_labels.loc[yes_ind]
    X2_test = X2_test.drop(['Prediction'], axis=1)

    # clean up the test_features dataset for future modeling, means remove the Prediction column
    test_features_temp = test_features_temp.drop(['Prediction'], axis=1)
    proba = gb_clf1.predict_proba(X2_test) 
    pred = gb_clf1.predict(X2_test) 

    # build the 2nd model to be used model later on the validate dataset

    #setup model parameters, change some of the defaults based on benchmarking
    gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=10, 
                                        max_depth=3, random_state = None, subsample = 1.0, criterion='mse', 
                                        min_samples_split = 10, min_samples_leaf = 10) # random_state=0, or 

    #since a false negative is much more likely than a false positive, we should weight them accordingly. 
    #IE Finding a true one is more important
    # note that the weights in the 2nd model are the inverse of the weights in the 1st model
    #gb_clf2.fit( train_features, train_labels, sample_weight=np.where(train_labels == 1,wt2_1,wt2_0) ) # was 0.1
    if (debug == 1):
        print('create M2 train_labels:', train_labels)
        print('create M2: train_labels\n', train_labels['Class'].value_counts())
        
        
    #gb_clf2.fit( train_features, train_labels, sample_weight=np.where(train_labels == 1,wt2_1,wt2_0) )
    gb_clf2.fit( train_features, train_labels, sample_weight=np.where(train_labels == 1, wt2_1, wt2_0) )

    
    #use model to predict validation dataset
    predictions = gb_clf2.predict(X2_test) 
    return gb_clf1, gb_clf2

def run_model(gb_clf1, gb_clf2, train_features, train_labels, test_features, test_labels, val_features, val_labels, algo, rn1, rn2, spl1, spl2, wt1_1, wt1_0, wt2_1, wt2_0):

    #-----------------------------------------------------------------------------------------------------
    # Now that we have built the 2 models from the test dataset, run the untouched validate dataset through both of them to get an unbiased result to compare against

    # run the validate dataset through the first model
    if (debug == 1):
        print('Run M1 val_labels:',val_labels)
        print('Run M1: val_labels\n', val_labels['Class'].value_counts())
    predictions1 = gb_clf1.predict(val_features)
    predictions_proba1 = gb_clf1.predict_proba(val_features)
    X1_val_final = val_features.copy()
    X1_val_final=X1_val_final.join(val_labels)
    X1_val_final['Proba_1'] = predictions_proba1[:,1]

    tn1, fp1, fn1, tp1 = confusion_matrix( val_labels, predictions1).ravel()
    # use both models to predict final validation dataset
    
    val_features_temp = val_features.copy()
    val_features_temp['Prediction'] = predictions1

    yes_ind = val_features_temp[val_features_temp['Prediction'] == 1].index

    X2_val = val_features_temp.loc[yes_ind]
    y2_val = val_labels.loc[yes_ind]
    X2_val = X2_val.drop(['Prediction'], axis=1)
    # run the validate dataset through the second model
    if (debug == 1):
        print('Run M2 y2_val:', y2_val)
        print('Run M2: y2_val\n', y2_val['Class'].value_counts())
    predictions2 = gb_clf2.predict(X2_val)

    X2_val_final = X2_val.copy()
    X2_val_final.join(y2_val)
    predictions_proba2 = gb_clf2.predict_proba(X2_val)
    # validate the join!!
    X2_val_final['Proba_2'] = predictions_proba2[:,1]
    X2_val_final

    cols_to_use = X2_val_final.columns.difference(X1_val_final.columns)
    val_features_final = X1_val_final.join(X2_val_final[cols_to_use], how='left', lsuffix='_1', rsuffix='_2')
    # rowwise action (axis=1)
    val_features_final.loc[val_features_final['Proba_2'].isnull(),'Proba_2'] = val_features_final['Proba_1']
    #val_features_final.query("Proba_1 != Proba_2")

    tn2, fp2, fn2, tp2 = confusion_matrix(y2_val, predictions2).ravel()

    return tn1, fp1, fn1, tp1, tn2, fp2, fn2, tp2

def get_dataset():
    colab = os.environ.get('COLAB_GPU', '10')
    if (int(colab) == 0):
        from google.colab import drive
        drive.mount('/content/drive')  
    else:
        print("")

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

    #df = pd.read_csv('creditcard.csv')
    df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
    return df

def split_into_3(X, y, spl1, spl2, rn1, rn2):
    # Split the Data into Train and Test 

    # create train, test and validate datasets
    # get repeatable list of "random" numbers for reproducability of your code
    # set.seed(x)
    # first split original into Train and Test+Val
    #rn1=42
    #rn2=4
    # not used: X_test1, y_test1 -> they are temporary, to be split apart in the 2nd train_test_split call 
    X_train, X_test1, y_train, y_test1 = train_test_split(X,y, test_size = spl1, random_state = None, shuffle=True)
    # then split Test+Val into Test and Validate
    # Validate will only be used in the 2 Model system (explained below)
    X_test, X_val, y_test, y_val = train_test_split(X_test1,y_test1, test_size = spl2, random_state = None, shuffle=True)
    class_names=[0,1] # name  of classes 1=fraudulent transaction
    
    # find the number of minority (value=1) samples in our train set so we can down-sample the majority of our train data to it
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
    
    a='''
    # do the same for the test data -- first test seems to indicate this makes things worse
    # find the number of minority (value=1) samples in our test set so we can down-sample the majority of our test data to it
    yes = len(y_test[y_test['Class'] ==1])
    # retrieve the indices of the minority and majority samples 
    yes_ind = y_test[y_test['Class'] == 1].index
    no_ind = y_test[y_test['Class'] == 0].index
    # random sample the majority indices based on the amount of 
    # minority samples
    new_no_ind = np.random.choice(no_ind, yes, replace = False)
    # merge the two indices together
    undersample_ind = np.concatenate([new_no_ind, yes_ind])
    # get undersampled dataframe from the merged indices of the test dataset
    X_test = X_test.loc[undersample_ind]
    y_test = y_test.loc[undersample_ind]'''
   
    return X_train, y_train, X_test, y_test, X_val, y_val

def analyze(df,X,y,min_,max_,compare, no_runs):
    cm_matrix_array = []
    for k in range(0, no_runs, 1):
        print('k', k)
        cm_results = []
        cm_matrix = []
        #for i in range(0, 9, 1):
        for i in np.linspace(min_, max_, num=8):
            for j in np.linspace(min_, max_, num=8):
                rn1=random.randint(1, 10)
                rn2=random.randint(1, 10)
                if (compare == 'split'):
                    spl1 = i/10
                    spl2 = j/10
                    wt1_1 = 11.0
                    wt1_0 = 1.0
                    wt2_1 = 3.0
                    wt2_0 = 1.0
                elif (compare == 'weight'): # compare between gb_clf1 and gb_clf2
                    spl1 = 0.3
                    spl2 = 0.3
                    wt1_1 = (2*i) - 1
                    wt1_0 = 1.0
                    wt2_1 = (2*j) - 1  
                    wt2_0 = 1.0
                elif (compare == 'weight1'): # compare within gb_clf1
                    spl1 = 0.3
                    spl2 = 0.3
                    wt1_1 = (2*i) - 1
                    wt1_0 = (2*j) - 1 
                    wt2_1 = 1.0 
                    wt2_0 = 1.0
                elif (compare == 'weight2'): # compare within gb_clf2
                    spl1 = 0.3
                    spl2 = 0.3
                    wt1_0 = 1.0
                    wt1_1 = 1.0
                    wt2_0 = (2*j) - 1  
                    wt2_1 = (2*i) - 1
                elif (compare == 'wt_ratio'): # compare within gb_clf2
                    spl1 = 0.3
                    spl2 = 0.3
                    wt1_1 = i
                    wt1_0 = max_ - wt1_1 + 1
                    wt2_1 = j 
                    wt2_0 = max_ - wt2_1 + 1 
                    if (debug == 1):
                        print('weights:', i, j, wt1_1, wt1_0, wt2_1, wt2_0, wt1_1/wt1_0, wt2_1/wt2_0)
                elif (compare == 'none'):
                    spl1 =  0.30
                    spl2 =  0.30
                    wt1_1 = 4.0
                    wt1_0 = 1.0
                    wt2_1 = 2.3
                    wt2_0 = 2.7
                else:
                    print("default compare is none")
                    spl1  =  0.3
                    spl2  =  0.3
                    wt1_1 = 11.0
                    wt1_0 =  1.0
                    wt2_1 =  3.0
                    wt2_0 =  1.0
                algo = 'ratio'           
                X_train, y_train, X_test, y_test, X_val, y_val = split_into_3(X, y, spl1, spl2, rn1, rn2)
                gb_clf1, gb_clf2 = create_models(X_train, y_train, X_test, y_test, X_val, y_val, algo, rn1, rn2, spl1, spl2, wt1_1, wt1_0, wt2_1, wt2_0) 
                tn1, fp1, fn1, tp1, tn2, fp2, fn2, tp2 = run_model(gb_clf1, gb_clf2, X_train, y_train, X_test, y_test, X_val, y_val, algo, rn1, rn2, spl1, spl2, wt1_1, wt1_0, wt2_1, wt2_0 )

                total = tn1+tn2+fp2+fn1+fn2+tp2
                sp = round((tn1 + tn2)/(tn1 + tn2 +fp2), 3)
                se = round(tp2/(tp2 + fn1 + fn2), 3)
                #print('results:', tn1, fp1, fn1, tp1, tn2, fp2, fn2, tp2 )
                cm_results.append([algo, 'step1','','','','', tn1, fp1, fn1, tp1, sp, se])
                cm_results.append([algo, 'step2','','','','', tn2, fp2, fn2, tp2, sp, se])
                cm_results.append([algo, 'final**', rn1, rn2, spl1, spl2, (tn1+tn2), fp2, (fn1+fn2), tp2, round((tn1+tn2)/total,3), round(fp2/total,3), round((fn1+fn2)/total,4), round(tp2/total,3), sp, se])
                row_result = [wt1_1, wt1_0, wt2_1, wt2_0, rn1, rn2, spl1, spl2, (tn1+tn2), fp2, (fn1+fn2), tp2, round((tn1+tn2)/total,3), round(fp2/total,3), round((fn1+fn2)/total,4), round(tp2/total,3), sp, se]
                cm_matrix.append(row_result)
        cm_matrix_array.append(cm_matrix)
        
    # print settings
    print('test, val, split settings')
    print(spl1, spl2)
    print('test, val, split sizes')
    print( (spl1-spl1*spl2), (spl1*spl2) )
    print('weight 1_1, weight 1_0, weight 2_1, weight 2_0')
    print(wt1_1, wt1_0, wt2_1, wt2_0)
    
    return cm_matrix_array

def plot_data(cm_matrix_array, x_label, y_label, z_label, z1_label, z2_label):
    if (debug == 1):
        print('plot_data')
        print(cm_matrix_array, x_label, y_label, z_label, z1_label, z2_label)
    # draw combined contour map
    cnt = np.array(cm_matrix_array).shape[0]
    rows = np.array(cm_matrix_array).shape[1]
    cols = np.array(cm_matrix_array).shape[2]
    if (debug == 1):
        print('cm_matrix_array')
        print(cnt, rows, cols)
    cm_matrix_sum = np.zeros((rows,cols))
    for n in range(0,cnt):
        cm_matrix_sum = cm_matrix_sum + np.array(cm_matrix_array[n])
    if (debug == 1):
        print(cm_matrix_sum/cnt)

    df_sum = pd.DataFrame( (cm_matrix_sum/cnt) ,columns=('wt1_1', 'wt1_0', 'wt2_1','wt2_0','rn1','rn2','spl1','spl2','TN','FP','FN','TP','TN_Pct','FP_Pct','FN_Pct','TP_Pct', 'SP', 'SE'))
    # add an average (SE + SP) metric
    df_sum['Average'] = (df_sum['SP'] + df_sum['SE'])/2
    df_sum['wt_ratio_1'] = (df_sum['wt1_1']/df_sum['wt1_0'])
    df_sum['wt_ratio_2'] = (df_sum['wt2_1']/df_sum['wt2_0'])
    # save output to a file for later detailed comparisons
    df_sum.to_csv ('export_dataframe.csv', index = False, header=True)

    # prepare df for contour map to find sweet spot
    # remove extra columns not used in comparison
    df_sum_in = df_sum.drop(columns=['rn1','rn2','TN','FP','FN','TP','TN_Pct','TP_Pct'])
    # draw contour map
    print("combined contour maps")

    #x=df_sum_in[x_label]
    #y=df_sum_in[y_label]
    if (x_label != ''):
        draw_contour(df_sum_in, x_label, y_label, z1_label)
        draw_contour(df_sum_in, x_label, y_label, z2_label)
        draw_contour(df_sum_in, x_label, y_label, z_label)
    else:
        print('not drawing contours as the x and y data are constant')
    df_final = df_sum.drop(columns=['rn1','rn2','spl1','spl2','TN','FP','FN','TP','TN_Pct','TP_Pct','FN_Pct','FP_Pct','wt_ratio_1','wt_ratio_2'])
    return df_final

def main():
    # read the full dataset from input file
    df = get_dataset()
    # divide full dataset into features and labels
    X = df.loc[:, df.columns != 'Class']
    y = df.loc[:, df.columns == 'Class']
    if (debug == 1):
        print('full dataset:',y)
        print('full dataset\n', df['Class'].value_counts())
    
    # setup parameters for optimization
    cm_matrix_array = []
    min_ = 1
    max_ = 4
    no_runs = 20
    compare = 'none'
    if (compare == 'split'):       
        x_label = 'spl1'
        y_label = 'spl2'
    elif (compare == 'weight'): # compare between gb_clf1 and gb_clf2
        x_label = 'wt1_1'
        y_label = 'wt2_1'
    elif (compare == 'weight1'): # compare within gb_clf1
        x_label = 'wt1_1'
        y_label = 'wt1_0'
    elif (compare == 'weight2'): # compare within gb_clf2
        x_label = 'wt2_1'
        y_label = 'wt2_0'
    elif (compare == 'wt_ratio'): # compare within gb_clf2
        x_label = 'wt_ratio_1'
        y_label = 'wt_ratio_2'
    elif (compare == 'none'):
        x_label = ''
        y_label = ''
        print('not changing any parameters')
    z1_label = 'SP'
    z2_label = 'SE'
    z_label = 'Average'

    
    # run model with range of parameters to find optimal settings
    # using a 2 dimensional contour map to compare 2 variables at a time
    # for example, compare test and validate split sizes or compare weights for 1st and 2nd step of model
    # parameters:
    #    weight:                 1 - 20
    #    splits:                 .1 - .9
    #    learning_rate:          .001 - .1
    #    n_estimators:           20-100 (larger is usually better)
    #    loss:                   deviance or exponential or my own function
    #    min_samples_per_split:  2-10
    #    min_samples_per_leaf:   1-10
    #    max_depth:              1-5
    
    # run the analysis
    cm_matrix_array = analyze(df, X, y, min_, max_, compare, no_runs)
        
    # plot the model output
    df_out = plot_data(cm_matrix_array, x_label, y_label, z_label, z1_label, z2_label)
    print('df_out')
    print(df_out)
    
if __name__ == "__main__":
    main()
