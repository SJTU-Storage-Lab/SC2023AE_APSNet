#conding=utf8  
import os
import re
from collections import Counter
import csv
import pandas as pd

column_list_MC1 = ["2","6","7","8","11","13","177","181","182","190","191","192","193","200","204","205","233","241","242","244","175","232"]

column_list_MC2 = ["2","6","7","8","11","13","177","181","182","190","191","192","193","200","211","204","205","233","241","242","244","175","232"]
g = os.walk(r"./output/")  

def is_in(full_str, sub_str):
    if re.findall(sub_str, full_str):
        return True
    else:
        return False

list = []

for path,dir_list,file_list in g:  
    for file_name in file_list:  
        if is_in(str(os.path.join(path, file_name)), "MC1_R") is True:
            list.append(os.path.join(path, file_name))

for i in range(len(list)):
    filename = list[i]
    print("now open file: ",filename)

    df = pd.read_csv(filename)
    d = df.drop(column_list_MC1,axis=1,inplace=False)
    d.to_csv(filename,index=False)
