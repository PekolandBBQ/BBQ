import os
import time

env=os.environ

if 'MY_POD_NAME' in env:
    id=os.environ['MY_POD_NAME']


now=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
target='/home/ai/workspace/app/output'+'/'+id+'/models'+'/'+id+'_'+'<'+now+'>'+'.model.pkl'
print(target)