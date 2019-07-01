
# coding: utf-8

# In[1]:

import numpy as np
np.random.seed(1991)
import datetime
import os
import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization


# In[2]:

def to_categorical(y, nb_classes=None):
    
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


# In[3]:

def batch_generator(X, y, batch_size, shuffle):
    
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


# In[4]:

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0


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
    f.write('id,label\n')
    total = 0
    test_val = test['id'].values
    for i in range(len(test_val)):
        str1 = str(test_val[i])
        for j in range(12):
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


# In[9]:

def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH,path_user_basic_info=USER_BASIC_INFO_PATH,
              path_user_behavior_info=USER_BEHAVIOR_INFO_PATH):

    basic = pd.read_csv(path_user_basic_info, names = ['uId','gender', 'city', 'prodName', 'ramCapacity', 
                                'ramLeftRation','romCapacity', 'romLeftRation', 'color', 'fontSize', 
                                'ct', 'carrier', 'os'],dtype={'uId': np.str})
    basic.drop_duplicates('uId', keep='first', inplace=True)
    
    basic['gender'] = pd.factorize( basic['city'],sort=True)[0]
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
    
   
    train_loader = pd.read_csv(path_train, names = ['uId','age_group'],dtype={'uId': np.str})
    train = train_loader
    train['age_group'] = pd.factorize( train['age_group'],sort=True)[0] 
    train = pd.merge(train, basic, how='left', on='uId', left_index=True )
    # target
    target = train.age_group
   
    train.drop(['uId','age_group'],axis =1 , inplace = True)
    train.fillna(-1, inplace=True)
    #train = pd.merge(train, events_small, how='left', on='device_id', left_index=True)


    test_loader = pd.read_csv(path_test, names = ['uId'], dtype={'uId': np.str})
    test = pd.merge(test_loader, basic, how='left', on='uId', left_index=True)
    test.drop('uId',axis =1 , inplace = True)
    test.fillna(-1, inplace=True)

 

    return train,test,target


# In[10]:

train,test,y =load_data()


# In[11]:

## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []
features = ['gender', 'city', 'prodName', 'ramCapacity', 
            'ramLeftRation', 'romCapacity', 'romLeftRation', 
            'color', 'fontSize', 'ct', 'carrier', 'os']

for f in features:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)



#del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

#del(xtr_te, sparse_data, tmp)


# In[12]:

## neural net
def nn_model():
    model = Sequential()
    model.add(Dense(1200, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(500, init = 'he_normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(6, init = 'he_normal',activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['accuracy'])
    return(model)


# In[13]:

## cv-folds
cv = StratifiedKFold(n_splits = 5, random_state=0)

## train models
i = 0
nbags = 20
nepochs = 10
number_class=y.nunique()

pred_oob = np.zeros((xtrain.shape[0],number_class))
pred_test = np.zeros((xtest.shape[0],number_class))

#target to categorical
y_cat=to_categorical(y.values)


# In[ ]:

#start training
for (inTr, inTe) in cv.split(xtrain,y):
    xtr = xtrain[inTr]
    ytr = y_cat[inTr]
    xte = xtrain[inTe]
    yte = y_cat[inTe]
    pred = np.zeros((xte.shape[0],number_class))
    for j in range(nbags):
        model = nn_model()
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 34, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0],
                                  verbose = 1,validation_data=(xte.todense(),yte))
        pred += model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])
        pred_test += model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])
    pred /= nbags
    pred_oob[inTe] = pred
    
    score = accuracy_score(yte, pred)
    
    i += 1
    print('Fold ', i, '- accuracy:', score)
    
total_score = accuracy_score(y, pred_oob)    

pred_test /= (nfolds*nbags)

print('Total - - accuracy:', total_score)


# In[ ]:

keras_predictions_test = pd.DataFrame(pred_test)
keras_prediction_train = pd.DataFrame(pred_oob)

keras_predictions_test.to_csv('nn1_predictions_test.csv',index=None)
keras_prediction_train.to_csv('nn1_prediction_train.csv',index=None)

