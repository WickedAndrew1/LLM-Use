import os
import glob
import pandas as pd
e=""
#Use file directory of folder with deepeval csv files
ee=os.path.abspath(e)
files=glob.glob(ee + '/*.csv')
sorted_files = sorted(files, key=lambda x: int(os.path.basename(x).split('.')[0]))
tc={}
s=0
c=0
p=0
a=0
for i,file, in enumerate(sorted_files):
    df=pd.read_csv(file)
    st="test_case_"
    for i, row in df.iterrows():
        st=st+str(a)
        stat=row["Status"]
        if 'Passed' in stat:
            s+=1
        con = row["Correctness (GEval) Success"]
        prom=row["Prompt Alignment (GEval) Success"]
        md={
            "Status": stat,
            "Prompt Alignment": prom,
            "Correctness": con,
        }
        if con==True:
            c+=1
        if prom==True:
            p+=1
        tc[st]=md
        a+=1
        st="test_case_"
d=[]
for i,met in tc.items():
    row={
        "Test Case": i,
        "Status": met["Status"],
        "Total Pass": s,
        "Total Correct": c,
        "Total Aligned": p
    }
#Somewhat lazy; I did not feel like creating one individual row for total number of passes, coherency, accuracy, and relevancy. It's inputted for each row.
    d.append(row)
odf=pd.DataFrame(d)
filename=os.path.join(ee,"RAG-Bool3-Full Trial 6 Results.csv")
odf.to_csv(filename, index=False)