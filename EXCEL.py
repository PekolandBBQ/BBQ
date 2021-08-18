import io
import pandas as pd
from pandas.core.algorithms import mode
from pandas.core.frame import DataFrame
import os
import numpy as np
from multiprocessing import Process
from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import RandomUnderSampler
# from imblearn.ensemble import EasyEnsemble

# df1 = pd.DataFrame({'V1':np.random.rand(100),
#                     'V2 ':np.random.rand(100),
#                     'V3':np.random.rand(100)})
# newdf=pd.DataFrame()
# newdf.to_excel(r'C:\Users\c3574\Desktop\demo.xlsx')
def getonehot(xlspath,sheetnum,workbook):

    st=pd.read_excel(io=xlspath,sheet_name='Xclsalllabelpurenum')
    # stcolumnsname=list(st)
    st_onehot=st.merge(pd.get_dummies(st))
    dfst=DataFrame(st_onehot)
    dfst.to_csv(r'C:\Users\c3574\Desktop\onehot.txt',sep='\t')
    # with pd.ExcelWriter(workbook,mode='a') as writer:
    #     dfst.to_excel(writer,sheet_name=sheetnum)
    #     writer.save
    print(dfst)

if __name__ == '__main__':
# # # st=st.join(pd.get_dummies(pd.get_dummies(st.受理日类型)))
    p1=Process(target=getonehot,args=(r"C:\Users\c3574\Desktop\label.xlsx",'onehot1',r'C:\Users\c3574\Desktop\onehot.xlsx',)) #必须加,号 
    # p3=Process(target=getonehot,args=(r"C:\Users\c3574\Desktop\test2.xlsx",'sheet6',r'C:\Users\c3574\Desktop\demo1.xlsx',))
    # p4=Process(target=getonehot,args=(r"C:\Users\c3574\Desktop\test3.xlsx",'sheet7',r'C:\Users\c3574\Desktop\demo1.xlsx',))

    p1.start()
    # p3.start()
    # p4.start()
    print('主线程')