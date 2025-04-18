import os
import glob
import pandas as pd
e=""
ee=os.path.abspath(e)
files=glob.glob(ee + '/*.csv')
sorted_files = sorted(files, key=lambda x: int(os.path.basename(x).split('.')[0]))
tc={}
s=0
su=0
re=0
c=0
a=0
for i,file, in enumerate(sorted_files):
    df=pd.read_csv(file)
    st="test_case_"
    for i, row in df.iterrows():
        st=st+str(a)
        stat=row["Status"]
        sum= row["Summarization Success"]
        rel= row["Answer Relevancy Success"]
        coh = row["Coherence (GEval) Success"]
        if 'Passed' in stat:
            s+=1
        if sum==True:
            su+=1
        if rel==True:
            re+=1
        if coh==True:
            c+=1
        md={
            "Status": stat,
            "Summary": sum,
            "Relevancy": rel,
            "Coherency": coh,
        }
        tc[st]=md
        a+=1
        st="test_case_"
d=[]
for i,met in tc.items():
    row={
        "Test Case": i,
        "Status": met["Status"],
        "Summary": met["Summary"],
        "Relevancy": met["Relevancy"],
        "Coherency": met["Coherency"],
        "Total Pass": s,
        "Total Summary": su,
        "Total Relevancy": re,
        "Total Coherency": c
    }
#Somewhat lazy; I did not feel like creating one individual row for total number of passes, coherency, accuracy, and relevancy. It's inputted for each row.
    d.append(row)
odf=pd.DataFrame(d)
filename=os.path.join(ee,"RAG-Summary1-Full Trial 2 Results.csv")
odf.to_csv(filename, index=False)