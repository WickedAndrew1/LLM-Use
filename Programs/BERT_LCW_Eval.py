import os
from csv import DictWriter
from evaluate import load
bertscore = load("bertscore")
from datasets import load_dataset
from huggingface_hub import login
login(token="") #Read Token.
dataset=load_dataset('adiaz21/LCW-Summary-3',split='train')
candidatetext=[]
referencetext=[]
for i in dataset:
    for j in i:
        if j=='answer':
            candidatetext.append(i[j])
        if j=='ground_truth':
            referencetext.append(i[j])
c=len(candidatetext)
r=len(referencetext)
a=0
data=[]
if c==r:
    results = bertscore.compute(predictions=candidatetext, references=referencetext, lang="en")
while a!=c:
    ct=candidatetext[a]
    rt=referencetext[a]
    precision=round(results['precision'][a],4)
    recall=round(results['recall'][a],4)
    f1=round(results['f1'][a],4)
    row={
        "Candidate Text": ct,
        "Reference Text": rt,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
    data.append(row)
    a+=1
ee=""
ex=os.path.abspath(ee)
#if os.path.exists(ex):
    #print(True)
d="BERT_LCW_Summary_Trial3.csv"
loc=ee+d
with open(loc,'w',newline='') as file:
    fieldnames=data[0].keys()
    writ=DictWriter(file,fieldnames=fieldnames)
    writ.writeheader()
    writ.writerows(data)
