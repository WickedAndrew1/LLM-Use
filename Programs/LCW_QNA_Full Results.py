import os
import glob
import pandas as pd
e=""
ee=os.path.abspath(e)
t=os.path.exists(ee)
files=glob.glob(ee + '/*.csv')
sorted_files = sorted(files, key=lambda x: int(os.path.basename(x).split('.')[0]))
tc={}
s=0
c=0
cc=0
ccc=0
a=0
for i,file, in enumerate(sorted_files):
    df=pd.read_csv(file)
    st="test_case_"
    for i, row in df.iterrows():
        st=st+str(a)
        stat=row["Status"]
        coh = row["Coherence (GEval) Success"]
        acc= row["Accuracy (GEval) Success"]
        rel= row["Answer Relevancy Success"]
        if 'Passed' in stat:
            s+=1
        if coh==True:
            c+=1
        if acc==True:
            cc+=1
        if rel==True:
            ccc+=1
        md={
            "Status": stat,
            "Coherency": coh,
            "Accuracy": acc,
            "Relevancy": rel
        }
        tc[st]=md
        a+=1
        st="test_case_"
d=[]
for i,met in tc.items():
    row={
        "Test Case": i,
        "Status": met["Status"],
        "Coherency": met["Coherency"],
        "Accuracy": met["Accuracy"],
        "Relevancy": met["Relevancy"],
        "Total Pass": s,
        "Total Coherency": c,
        "Total Accuracy": cc,
        "Total Relevancy": ccc
    }
#Somewhat lazy; I did not feel like creating one individual row for total number of passes, coherency, accuracy, and relevancy. It's inputted for each row.
    d.append(row)
odf=pd.DataFrame(d)
filename=os.path.join(ee,"LCW-Qna3-Full Trial 6 Results.csv")
odf.to_csv(filename, index=False)
    