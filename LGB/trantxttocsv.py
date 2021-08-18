import numpy as np
import pandas as pd
txt=np.loadtxt('LGB/data/labelresult1.txt',encoding='utf-8',skiprows=1)
txtDF=pd.DataFrame(txt)
txtDF.to_csv('LGB/data/labelresult1.csv',index=False,encoding='utf-8_sig')  