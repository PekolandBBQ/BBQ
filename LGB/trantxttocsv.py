import numpy as np
import pandas as pd
txt=np.loadtxt('../data/labelresultunder.txt',encoding='utf-8',skiprows=1)
txtDF=pd.DataFrame(txt)
txtDF.to_csv('../data/labelresultunder.csv',index=False,encoding='utf-8_sig')  