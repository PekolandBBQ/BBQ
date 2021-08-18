import io
import pandas as pd
from pandas.core.algorithms import mode
from pandas.core.frame import DataFrame
import os
import numpy as np
from multiprocessing import Process

# df1 = pd.DataFrame({'V1':np.random.rand(100),
#                     'V2 ':np.random.rand(100),
#                     'V3':np.random.rand(100)})
# newdf=pd.DataFrame()
# newdf.to_excel(r'C:\Users\c3574\Desktop\demo.xlsx')

st=pd.read_excel(io=r"C:\Users\c3574\Desktop\研究2 70W（全特征-删除样本）整理\受理日类型onehot-value-label.xlsx",sheet_name='label')
stcolumnsname=list(st)[1]
st_onehot=st.merge(pd.get_dummies(st,columns=[stcolumnsname]),on='工作单编号')
dfst=DataFrame(st_onehot)
with pd.ExcelWriter(r'C:\Users\c3574\Desktop\demo.xlsx',mode='a') as writer:
    dfst.to_excel(writer,sheet_name='Sheee')
    writer.save
    writer.close
print(dfst)