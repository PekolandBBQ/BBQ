import io
import pandas as pd
from pandas.core.algorithms import mode
from pandas.core.frame import DataFrame
import os
import numpy as np
from multiprocessing import Process
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
# from imblearn.over_sampling import RandomUnderSampler
# from imblearn.ensemble import EasyEnsemble

# df1 = pd.DataFrame({'V1':np.random.rand(100),
#                     'V2 ':np.random.rand(100),
#                     'V3':np.random.rand(100)})
# newdf=pd.DataFrame()
# newdf.to_excel(r'C:\Users\c3574\Desktop\demo.xlsx')
def getvalueover(xlspath):

    st=pd.read_excel(io=xlspath,sheet_name='Xclsallvaluepurenum')
    # col=list(st.columns)
    # # st[col]=st[col].apply(pd.to_numeric,errors='coerce').fillna(0.0)
    # # st=pd.DataFrame(st,dtype='float')
    x=st.iloc[:,:-1]
    print(x)
    y=st.iloc[:,-1]
    print(y)
    gdo=st.groupby('工单风险类别').count()
    print(gdo)
    smo=SMOTE(random_state=42)
    xsmo,ysmo=smo.fit_resample(x,y)
    print(type(xsmo))
    print(type(ysmo))
    # print(xsmo)
    # print(ysmo)
    xsmo=pd.DataFrame(xsmo)
    smore=pd.concat([xsmo,ysmo],axis=1,ignore_index=False)
    gdp=smore.groupby('工单风险类别').count()   
    print(gdp)
    smore.to_csv(r'C:\Users\c3574\Desktop\valueresult1.txt',sep='\t')
    # with pd.ExcelWriter(workbook,mode='a') as writer:
    #     smore.csv(writer,sheet_name=sheetnum)
    #     writer.save
    print(smore)
    
    # stcolumnsname=list(st)[1]
    # st_onehot=st.merge(pd.get_dummies(st,columns=[stcolumnsname]),on='工作单编号')
    # dfst=DataFrame(st_onehot)
    # with pd.ExcelWriter(workbook,mode='a') as writer:
    #     dfst.to_excel(writer,sheet_name=sheetnum)
    #     writer.save
    # print(dfst)
def getlabelover(xlspath):

    st=pd.read_excel(io=xlspath,sheet_name='Xclsalllabelpurenum')
    # col=list(st.columns)
    # # st[col]=st[col].apply(pd.to_numeric,errors='coerce').fillna(0.0)
    # # st=pd.DataFrame(st,dtype='float')
    x=st.iloc[:,:-1]
    print(x)
    y=st.iloc[:,-1]
    print(y)
    gdo=st.groupby('工单风险类别').count()
    print(gdo)
    smo=SMOTE(random_state=42)
    xsmo,ysmo=smo.fit_resample(x,y)
    print(type(xsmo))
    print(type(ysmo))
    # print(xsmo)
    # print(ysmo)
    xsmo=pd.DataFrame(xsmo)
    smore=pd.concat([xsmo,ysmo],axis=1,ignore_index=False)
    gdp=smore.groupby('工单风险类别').count()   
    print(gdp)
    smore.to_csv(r'C:\Users\c3574\Desktop\labelresult1.txt',sep='\t')
    # with pd.ExcelWriter(workbook,mode='a') as writer:
    #     smore.csv(writer,sheet_name=sheetnum)
    #     writer.save
    print(smore)

def getvalueunder(xlspath):

    st=pd.read_excel(io=xlspath,sheet_name='Xclsallvaluepurenum')
    # col=list(st.columns)
    # # st[col]=st[col].apply(pd.to_numeric,errors='coerce').fillna(0.0)
    # # st=pd.DataFrame(st,dtype='float')
    x=st.iloc[:,:-1]
    print(x)
    y=st.iloc[:,-1]
    print(y)
    gdo=st.groupby('工单风险类别').count()
    print(gdo)
    smo=RandomUnderSampler(random_state=42)
    xsmo,ysmo=smo.fit_resample(x,y)
    print(type(xsmo))
    print(type(ysmo))
    # print(xsmo)
    # print(ysmo)
    xsmo=pd.DataFrame(xsmo)
    smore=pd.concat([xsmo,ysmo],axis=1,ignore_index=False)
    gdp=smore.groupby('工单风险类别').count()   
    print(gdp)
    smore.to_csv(r'C:\Users\c3574\Desktop\valueresultunder.txt',sep='\t')
    # with pd.ExcelWriter(workbook,mode='a') as writer:
    #     smore.csv(writer,sheet_name=sheetnum)
    #     writer.save
    print(smore)
    
    # stcolumnsname=list(st)[1]
    # st_onehot=st.merge(pd.get_dummies(st,columns=[stcolumnsname]),on='工作单编号')
    # dfst=DataFrame(st_onehot)
    # with pd.ExcelWriter(workbook,mode='a') as writer:
    #     dfst.to_excel(writer,sheet_name=sheetnum)
    #     writer.save
    # print(dfst)
def getlabelunder(xlspath):

    st=pd.read_excel(io=xlspath,sheet_name='Xclsalllabelpurenum')
    # col=list(st.columns)
    # # st[col]=st[col].apply(pd.to_numeric,errors='coerce').fillna(0.0)
    # # st=pd.DataFrame(st,dtype='float')
    x=st.iloc[:,:-1]
    print(x)
    y=st.iloc[:,-1]
    print(y)
    gdo=st.groupby('工单风险类别').count()
    print(gdo)
    smo=RandomUnderSampler(random_state=42)
    xsmo,ysmo=smo.fit_resample(x,y)
    print(type(xsmo))
    print(type(ysmo))
    # print(xsmo)
    # print(ysmo)
    xsmo=pd.DataFrame(xsmo)
    smore=pd.concat([xsmo,ysmo],axis=1,ignore_index=False)
    gdp=smore.groupby('工单风险类别').count()   
    print(gdp)
    smore.to_csv(r'C:\Users\c3574\Desktop\labelresultunder.txt',sep='\t')
    # with pd.ExcelWriter(workbook,mode='a') as writer:
    #     smore.csv(writer,sheet_name=sheetnum)
    #     writer.save
    print(smore)
def getvaluesvm(xlspath):

    st=pd.read_excel(io=xlspath,sheet_name='Xclsallvaluepurenum')
    # col=list(st.columns)
    # # st[col]=st[col].apply(pd.to_numeric,errors='coerce').fillna(0.0)
    # # st=pd.DataFrame(st,dtype='float')
    x=st.iloc[:,:-1]
    print(x)
    y=st.iloc[:,-1]
    print(y)
    gdo=st.groupby('工单风险类别').count()
    print(gdo)
    smo=SVC(class_weight='balanced')
    xsmo,ysmo=smo.fit(x,y)
    print(type(xsmo))
    print(type(ysmo))
    # print(xsmo)
    # print(ysmo)
    xsmo=pd.DataFrame(xsmo)
    smore=pd.concat([xsmo,ysmo],axis=1,ignore_index=False)
    gdp=smore.groupby('工单风险类别').count()   
    print(gdp)
    smore.to_csv(r'C:\Users\c3574\Desktop\valueresultsvm.txt',sep='\t')
    # with pd.ExcelWriter(workbook,mode='a') as writer:
    #     smore.csv(writer,sheet_name=sheetnum)
    #     writer.save
    print(smore)
    
    # stcolumnsname=list(st)[1]
    # st_onehot=st.merge(pd.get_dummies(st,columns=[stcolumnsname]),on='工作单编号')
    # dfst=DataFrame(st_onehot)
    # with pd.ExcelWriter(workbook,mode='a') as writer:
    #     dfst.to_excel(writer,sheet_name=sheetnum)
    #     writer.save
    # print(dfst)
def getlabelsvm(xlspath):

    st=pd.read_excel(io=xlspath,sheet_name='Xclsalllabelpurenum')
    # col=list(st.columns)
    # # st[col]=st[col].apply(pd.to_numeric,errors='coerce').fillna(0.0)
    # # st=pd.DataFrame(st,dtype='float')
    x=st.iloc[:,:-1]
    print(x)
    y=st.iloc[:,-1]
    print(y)
    gdo=st.groupby('工单风险类别').count()
    print(gdo)
    smo=SVC(class_weight='balanced')
    xsmo,ysmo=smo.fit(x,y)
    print(type(xsmo))
    print(type(ysmo))
    # print(xsmo)
    # print(ysmo)
    xsmo=pd.DataFrame(xsmo)
    smore=pd.concat([xsmo,ysmo],axis=1,ignore_index=False)
    gdp=smore.groupby('工单风险类别').count()   
    print(gdp)
    smore.to_csv(r'C:\Users\c3574\Desktop\labelresultsvm.txt',sep='\t')
    # with pd.ExcelWriter(workbook,mode='a') as writer:
    #     smore.csv(writer,sheet_name=sheetnum)
    #     writer.save
    print(smore)

if __name__ == '__main__':
# # # st=st.join(pd.get_dummies(pd.get_dummies(st.受理日类型)))
    # p1=Process(target=getvalueover,args=(r"C:\Users\c3574\Desktop\value.xlsx",)) 
    # p2=Process(target=getlabelover,args=(r"C:\Users\c3574\Desktop\label.xlsx",))#必须加,号 
    # p3=Process(target=getvalueunder,args=(r"C:\Users\c3574\Desktop\value.xlsx",)) 
    # p4=Process(target=getlabelunder,args=(r"C:\Users\c3574\Desktop\label.xlsx",))
    p5=Process(target=getvaluesvm,args=(r"C:\Users\c3574\Desktop\value.xlsx",)) 
    p6=Process(target=getlabelsvm,args=(r"C:\Users\c3574\Desktop\label.xlsx",))

    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    p5.start()
    p6.start()
    print('主线程')