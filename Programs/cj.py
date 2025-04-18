import json
def conjson(file):
    with open(file, 'r') as filename:
        data=json.load(filename)
    header1="ground_truth"
    header2="answer"
    for i in data:
        if header1 in i:
            i[header1]=str(i[header1])
            i[header1]=str.upper(i[header1])
        if header2 in i:
            i[header2]=str(i[header2])
            i[header2]=str.upper(i[header2])
    with open(file,'w') as filename:
        json.dump(data,filename)
if True:
    conjson("lcwb1.json")
    conjson("lcwb2.json")
    conjson("lcwb3.json")