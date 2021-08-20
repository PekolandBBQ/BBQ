import time
import pandas as pd
import joblib
import numpy as np
import os

#读取测试文本
test_data=pd.read_csv('test5.csv')

#加载模型
fList=os.listdir('LGB/model')
fList.sort(reverse=True)
model='LGB/model/'+fList[0]

#预测
clf=joblib.load(filename=model)
re=np.argmax(clf.predict(test_data),axis=1)
re=pd.DataFrame(re)
re[0]=re[0]+1

#保存结果
now=time.strftime("result_%Y%m%d%H%M%S", time.localtime())
saveresult='LGB/result/'+now+'.csv'
result_data=pd.concat([test_data,re],axis=1)
result_data.rename(columns={0:'预测工单类别'},inplace=True)
result_data.to_csv(saveresult)
