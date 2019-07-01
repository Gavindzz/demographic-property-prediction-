
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import gc


# In[2]:

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)


# In[3]:

class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


# In[4]:

def get_oof(clf):
    oof_train = np.zeros((ntrain,6))
    oof_test = np.zeros((ntest,6))
    

    for i, (train_index, test_index) in enumerate(kf.split(train,y_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test[:] += clf.predict(x_test)

    oof_test[:] /= NFOLDS
    return oof_train, oof_test


# In[5]:

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' %
              (tmin, round(tsec, 2)))


# In[6]:

def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


# In[7]:

def create_submission(score, prediction):
    # Make Submission
    test=pd.read_csv('/media/gavin/3fa6c02e-278d-413f-ae59-809a756c3966/huawei/age_test.csv',header=0)
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('uId,label\n')
    total = 0
    test_val = test['uId'].values
    for i in range(len(test_val)):
        str1 = str(test_val[i])
        for j in range(1):
            str1 += ',' + str(prediction[i][j])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


# In[8]:

DATA_TRAIN_PATH = '/media/gavin/3fa6c02e-278d-413f-ae59-809a756c3966/huawei/age_train.csv'
DATA_TEST_PATH = '/media/gavin/3fa6c02e-278d-413f-ae59-809a756c3966/huawei/age_test.csv'
USER_BASIC_INFO_PATH='/media/gavin/3fa6c02e-278d-413f-ae59-809a756c3966/huawei/user_basic_info.csv'
USER_BEHAVIOR_INFO_PATH='/media/gavin/3fa6c02e-278d-413f-ae59-809a756c3966/huawei/user_behavior_info.csv'
APP_INFO_PATH='/media/gavin/3fa6c02e-278d-413f-ae59-809a756c3966/huawei/app_info.csv'
USER_APP_ACTIVED_PATH='/media/gavin/3fa6c02e-278d-413f-ae59-809a756c3966/huawei/user_app_actived.csv'
USER_APP_USAGE_PATH='/media/gavin/3fa6c02e-278d-413f-ae59-809a756c3966/huawei/user_app_usage.csv'


# In[9]:

def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH,path_user_basic_info=USER_BASIC_INFO_PATH,
              path_user_behavior_info=USER_BEHAVIOR_INFO_PATH ,app_info_path=APP_INFO_PATH,
              user_app_avtived_path= USER_APP_ACTIVED_PATH,user_app_usage_path=USER_APP_USAGE_PATH):
    
    # User basic info

    basic=pd.read_csv(path_user_basic_info, header=0,names = ['uId','gender', 'city', 'prodName', 'ramCapacity', 
                                'ramLeftRation','romCapacity', 'romLeftRation', 'color', 'fontSize', 
                                'ct', 'carrier', 'os'],dtype={'uId': np.str})
    basic.drop_duplicates('uId', keep='first', inplace=True)
    
    basic['gender'] = pd.factorize( basic['gender'],sort=True)[0]
    basic['city'] = pd.factorize( basic['city'],sort=True)[0]
    basic['prodName'] = pd.factorize( basic['prodName'],sort=True)[0]
    basic['ramCapacity'] = pd.factorize( basic['ramCapacity'],sort=True)[0]
    basic['ramLeftRation'] = pd.factorize( basic['ramLeftRation'],sort=True)[0]
    basic['romCapacity'] = pd.factorize( basic['romCapacity'],sort=True)[0]
    basic['romLeftRation'] = pd.factorize( basic['romLeftRation'],sort=True)[0]
    basic['color'] = pd.factorize( basic['color'],sort=True)[0]
    basic['fontSize'] = pd.factorize( basic['fontSize'],sort=True)[0]
    basic['ct'] = pd.factorize( basic['ct'],sort=True)[0]
    basic['carrier'] = pd.factorize( basic['carrier'],sort=True)[0]
    basic['os'] = pd.factorize( basic['os'],sort=True)[0]
    
        
    train_loader = pd.read_csv(path_train,header=0, names = ['uId','age_group'],dtype={'uId': np.str})
    train = train_loader
    train['age_group'] = pd.factorize( train['age_group'],sort=True)[0] 
    
    
    #merge
    user_app_actived = pd.read_csv(user_app_avtived_path,header=0, names = ['uId','appId'],dtype={'uId': np.str})
    user_app_actived['appId'] = pd.factorize(user_app_actived['appId'],sort=True)[0]
    
    app_info = pd.read_csv(app_info_path, header=0,names = ['appId','category'])
    app_info['appId'] = pd.factorize(app_info['appId'],sort=True)[0]
    app_info['category'] = pd.factorize(app_info['category'],sort=True)[0]
 
    app=pd.merge(user_app_actived, app_info, how='left', on ='appId')
    add1 = pd.merge(basic, app, how='left', on ='uId')
    #train
    train = pd.merge(train ,add1, how='left',on ='uId')
    train.fillna(-1, inplace=True)
    
    
    # target
    target = train.age_group
   
    train.drop(['uId','age_group'],axis =1 , inplace = True)
    train.fillna(-1, inplace=True)

    test_loader = pd.read_csv(path_test, names = ['uId'],dtype={'uId': np.str})
    test = pd.merge(test_loader, add1, how='left',on='uId', left_index=True)
    test.drop_duplicates('uId' , keep ='first' , inplace =True )
    test.drop('uId',axis =1 , inplace = True)
    test.fillna(-1, inplace=True)
    
    
    return train,test,target


# In[ ]:

train , test , y_train = load_data()
gc.collect()
lable_group = LabelEncoder()
Y = lable_group.fit_transform(y_train)

NFOLDS = 5
SEED = 0

print("{},{}".format(train.shape, test.shape))


# In[ ]:

x_train = train.values
ntrain=train.shape[0]
x_test = test.values
ntest=test.shape[0]

kf =StratifiedKFold(n_splits = NFOLDS,random_state=SEED)


et_params = {
    'n_jobs': 2,
    'n_estimators': 600,
    'max_features': 0.5,
    'max_depth': 7,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 2,
    'n_estimators': 600,
    'max_features': 0.4,
    'max_depth': 7,
    'min_samples_leaf': 2,
}

et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

#start training the RF and ET models 
et_oof_train, et_oof_test = get_oof(et)
rf_oof_train, rf_oof_test = get_oof(rf)


# print the CV results 

print("ET-CV: {}".format(log_loss(y_train, et_oof_train)))
print("RF-CV: {}".format(log_loss(y_train, rf_oof_train)))


# In[ ]:

# i concatenate the train prediction with the target value so to use it in the next step : Stacking which will be our final model.
x_train =np.concatenate((et_oof_train, rf_oof_train,pd.DataFrame( y_train)), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test), axis=1)


rf_et_predictions_test = pd.DataFrame(x_test)
rf_et_prediction_train = pd.DataFrame(x_train)

rf_et_predictions_test.to_csv('rf_et_predictions_test.csv',index=None)
rf_et_prediction_train.to_csv('rf_et_prediction_train.csv',index=None)

#print the resulting train and data set's shape 
print("{},{}".format(x_train.shape, x_test.shape))

