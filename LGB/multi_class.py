from lightgbm.engine import train
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import gc

## load data
data = pd.read_csv('LGB/data/labelresult1.csv')#读取数据
# print(data.columns)

del_lable = ['label']
features = [i for i in data.columns if i not in del_lable]
cate_feature = features
# print(features)
X_data = data[features]
y_data = data['label'].astype(int)
# print(y_data)
xtrain,xtest,ytrain,ytest=train_test_split(X_data,y_data,train_size=0.85,random_state=2021)
train=pd.concat([xtrain,ytrain])
test=pd.concat([xtest,ytest])
# print(xtrain.shape,xtest.shape,ytrain.shape,ytest.shape)

num_round = 1000

## category feature one_hot
# test_data['label'] = -1
# data = pd.concat([train_data, test_data])
# cate_feature = ['gender', 'cell_province', 'id_province', 'id_city', 'rate', 'term']
# for item in cate_feature:
#     data[item] = LabelEncoder().fit_transform(data[item])

# train = data[data['label'] != -1]
# test = data[data['label'] == -1]

##Clean up the memory
del  data
gc.collect()

## get train feature



params = {'num_leaves': 60,
          'min_data_in_leaf': 30,
          'objective': 'multiclass',
          'num_class':49,#类别数
          'max_depth': -1,
          'learning_rate': 0.03,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": 15,
          'metric': 'multi_logloss',
          "random_state": 2021,
          # 'device': 'gpu' 
          }


folds = KFold(n_splits=5, shuffle=True, random_state=2019)
prob_oof = np.zeros((xtrain.shape[0], 49))
test_pred_prob = np.zeros((xtest.shape[0], 49))

## train and predict
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train)):
    print("fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(xtrain.iloc[trn_idx], label=ytrain.iloc[trn_idx])
    val_data = lgb.Dataset(xtrain.iloc[val_idx], label=ytrain.iloc[val_idx])

    clf = lgb.train(params,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=20,
                    categorical_feature=cate_feature,
                    early_stopping_rounds=60)
    prob_oof[val_idx] = clf.predict(xtrain.iloc[val_idx], num_iteration=clf.best_iteration)


    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    test_pred_prob += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
result = np.argmax(test_pred_prob, axis=1)

## plot feature importance
cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=False).index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)].sort_values(by='importance',ascending=False)
plt.figure(figsize=(8, 10))
sns.barplot(y="Feature",
            x="importance",
            data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('../../result/lgb_importances.png')
