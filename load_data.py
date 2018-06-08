@Author: Will Suess
@Date:   2018-06-06 20:56:28 PM
@Email:  will.suess@cfraresearch.com
# @Last modified by:   Will Suess
# @Last modified time: 2018-06-08 05:21:40 AM
'''
=========================================================
Load Data
=========================================================
'''
#%%
import pandas as pd
import numpy as np
import os
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from pipe import FactorExtractor
from sklearn.metrics import SCORERS
SCORERS
#%%
os.getcwd()
input_dir = 'C:/users/suess/credit/data/'
filename_1 = 'application_train.csv'
train_path = os.path.join(input_dir,filename_1)
filename_2 = 'application_test.csv'
test_path = os.path.join(input_dir,filename_2)

#%%

%time train = pd.read_csv(train_path)
%time test = pd.read_csv(test_path)
print(train.shape,test.shape)
dfs = [test,train]

for d in dfs:
    d.columns = d.columns.str.lower()
#
# labels = pd.concat([train,test])
#
# contract_type = pd.get_dummies(labels.name_contract_type.str.lower())
# labels = pd.concat([labels,contract_type],axis=1)
#
# train_rows = train.shape[0]
# test_rows = test.shape[0]
# train = labels.iloc[0:train_rows,:]
# test = labels.iloc[train_rows:,:]
#
train.shape
test.shape
# train.head()
train.columns = train.columns.str.replace(' ','_')
test.columns = test.columns.str.replace(' ','_')
#
# train.iloc[0:10,0:10].info()
#
# feats=[ 'NAME_CONTRACT_TYPE',
#         'cash_loans',
#         'revolving_loans',
#         'CODE_GENDER',
#         'FLAG_OWN_CAR',
#         'FLAG_OWN_REALTY',
#         'CNT_CHILDREN',
#         'AMT_INCOME_TOTAL',
#         'AMT_CREDIT',
#         'AMT_ANNUITY',
#         'AMT_GOODS_PRICE',
#         'NAME_TYPE_SUITE',
#         'NAME_INCOME_TYPE',
#         'NAME_EDUCATION_TYPE',
#         'NAME_FAMILY_STATUS',
#         'NAME_HOUSING_TYPE']
# train[feats].info()
#
# feats = [i.lower() for i in feats]
# gender = pd.get_dummies(train['code_gender'].str.lower())
# loan_type = train[['cash_loans','revolving_loans']]
# inc_credit = train[['amt_credit','amt_income_total','amt_annuity']]
# c_rows = np.random.randint(0,307511,4999)
# t = train['target']
# s_train = pd.concat([gender,loan_type,inc_credit,t],axis=1)
# s_train_chart = s_train.iloc[c_rows]
#
#
# alt.Chart(s_train_chart,background='white').mark_point().encode(
#     x='amt_credit',
#     y = 'amt_income_total',
#     color='target')
#

# s_train.target.value_counts()/s_train.__len__()

#%%
# Train using some basic cross validation
quant_f = [
            'amt_credit','amt_income_total',
            'amt_annuity','cnt_children',
            'days_birth','days_employed',
            'days_registration','days_id_publish',
            'own_car_age', 'obs_30_cnt_social_circle',
            'def_30_cnt_social_circle','obs_60_cnt_social_circle',
            'def_60_cnt_social_circle']

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
clf_rf = RandomForestClassifier(
                                max_depth=None,
                                min_weight_fraction_leaf=0.01,
                                n_estimators=100,
                                n_jobs=-1,
                                random_state=100,
                                max_features="auto",
                                class_weight={0:9,1:1},
                                verbose=100)
pipeline = make_pipeline(
                FactorExtractor(quant_f),
                Imputer(strategy='median'),
                StandardScaler(copy=True, with_mean=True, with_std=True),
                clf_rf)
params = dict(
                randomforestclassifier__min_weight_fraction_leaf = [0.10,0.05,0.01,.001],
                # randormforestclassifier__n_estimators = [100,200,300,400,500,1000,1500],
                # randomforestclassifier__max_features=['sqrt','log2',None])

rf_gridcv = GridSearchCV(
                            estimator=pipeline,
                            param_grid=params,
                            cv=10,
                            refit=True,
                            verbose=100,
                            scoring='roc_auc',
                            n_jobs=-1)
rf_gridcv

rf_gridcv.fit(train.loc[:,quant_f],train.target)
restartclf_rf.get_params.keys()
rf_gridcv.best_score_
rf_gridcv.best_params_


clf.fit(X_train,t)
y_pred = clf.predict(X_train)
y_prob = clf.predict_proba(X_train)
np.array(t)
y_pred
from sklearn.metrics import classification_report

rpt = classification_report(y_true=t, y_pred=y_pred, labels=[1,0])
print(rpt)

fpr, tpr, _ = roc_curve(y_true=t, y_score=y_prob[:,1], pos_label=1)
auc = pd.DataFrame([fpr,tpr],index=['fpr','tpr']).transpose()
auc.__len__()
c_rows_2 = np.random.randint(0,20631,5000)
c_auc = auc.iloc[c_rows_2]
c_auc.__len__()
c_auc.loc[:,'x'] = np.linspace(0,1,5000)
c_auc.loc[:,'y'] = c_auc.x
auc = alt.Chart(auc.iloc[c_rows_2],background='white')
auc.mark_line().encode(
    x = 'fpr',
    y = 'tpr',
    x = 'x',
    y = 'y'   )
