
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
import gc


# In[ ]:

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' %
              (tmin, round(tsec, 2)))


# In[ ]:

def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


# In[ ]:

#------------- the stacked model: load all the predictions (6 model) and build on top of it an xgboost model-------------------------#

# load xgboost predctions 
xgb1_train = pd.read_csv('xgb1_predictions_train.csv',header=0)
xgb1_test =pd.read_csv('xgb1_predictions_test.csv',header=0)


# In[ ]:

#load xgboost predictions with no events

xgb2_train=pd.read_csv('xgb2_predictions_train.csv',header=0)
xgb2_test=pd.read_csv('xgb2_predictions_test.csv',header=0)


# In[ ]:

#load keras predictions with events : 
nn1_train=pd.read_csv('nn1_prediction_train.csv',header=0)
nn1_test_train=pd.read_csv('nn1_predictions_test.csv',header=0)


# In[ ]:

#load keras prediction with no events

nn2_test=pd.read_csv('nn2_test.csv',header=0)
nn2_train=pd.read_csv('nn2_train.csv',header=0)


# In[ ]:

#load extra tree end RF prediction on the data with events which also has the raget variable

rf_et_predictions_test=pd.read_csv('rf_et_predictions_test.csv',header=0)
rf_et_prediction_train=pd.read_csv('rf_et_prediction_train.csv',header=0)
columns = rf_et_prediction_train.columns.tolist()
Target_name = columns[len(columns)-1]
target= rf_et_prediction_train[Target_name]
rf_et_prediction_train.drop(Target_name,axis=1,inplace=True)


# In[ ]:

# concat all train and test data in the same ordere 

train=pd.concat((xgb1_train,xgb2_train,nn1_train,nn2_train,rf_et_prediction_train), axis = 1)
test=pd.concat((xgb1_test,xgb2_test,nn1_test,nn2_test,rf_et_predictions_test),axis=1)


# In[ ]:

################################## Actual Run Code ##################################
lable_group = LabelEncoder()
Y = lable_group.fit_transform(target)


# In[ ]:

# enter the number of folds from xgb.cv
ntest=test.shape[0]
folds = 5
early_stopping = 50
oof_test = np.zeros((ntest,12))
start_time = timer(None)


# In[ ]:

# Load data set and target values
test = test.loc[:,~test.columns.duplicated()]


# In[ ]:

d_test = xgb.DMatrix(test)


# In[ ]:

# set up KFold that matches xgb.cv number of folds
kf = StratifiedKFold(n_splits = folds,random_state=0)

#Start the CV
for i, (train_index, test_index) in enumerate(kf.split(train,Y)):
    print('\n Fold %d\n' % (i + 1))
    X_train, X_val = train.values[train_index], train.values[test_index]
    y_train, y_val = Y[train_index], Y[test_index]

#######################################
#
# Define  variables
#
#######################################

    params = {}
    params["booster"]= "gbtree"
    params['objective'] = "multi:softprob"
    params['eval_metric'] ='mloss'
    params['num_class']=12
    params['eta'] = 0.01
    params['gamma'] = 0.1
    params['min_child_weight'] = 1
    params['colsample_bytree'] = 0.5
    params['subsample'] = 0.8
    params['max_depth'] = 7  
    params['silent'] = 1
    params['random_state'] = 0

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(d_train, 'train'), (d_valid, 'eval')]

####################################
#  Build Model
####################################
    clf = xgb.train(params,
                    d_train,
                    100000,
                    watchlist,
                    early_stopping_rounds=early_stopping) 
    timer(start_time)
####################################
#  Evaluate Model and Predict
####################################

    oof_test[:] += clf.predict(d_test, ntree_limit=clf.best_iteration) 
    print(' eval-accuracy: %.6f' % accuracy_score( y_val, clf.predict(d_valid), ntree_limit=clf.best_iteration))


# In[ ]:

oof_test /= folds

####################################
#  Make a submision
####################################

result = pd.DataFrame(oof_test, columns=lable_group.classes_)
test=pd.read_csv('/media/gavin/3fa6c02e-278d-413f-ae59-809a756c3966/huawei/age_test.csv',header=0)
result["uId"] = test.uId
result = result.set_index("uId")
now = datetime.now()
sub_file = 'submission_' + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
result.to_csv(sub_file, index=True,index_label='uId')

