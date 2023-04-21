import os
import re
import csv
import pandas as pd

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

    x_indicator = 0
    x_record = []
    flag =0

    with open(filename) as infile:
        reader = csv.reader(infile)
        
        for row in reader:
            temp = row
            flag = 0
            for i in range(len(temp)):
                if i>1:
                    if str(temp[i])!='-1' and str(temp[i])!='' and str(temp[i])!=',\n':
                        flag = 1
                        break
            
            if flag == 0:
                x_record.append(x_indicator)
                
            x_indicator+=1

    print(len(x_record))

    infile.close()

    for i in range(len(x_record)):
        x_record[i]= x_record[i]-1
        
    df = pd.read_csv(filename)
    d=df.drop(x_record,inplace=False)
    d.to_csv(filename,index=False)  
