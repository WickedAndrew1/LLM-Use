import pandas as pd
df1=pd.read_json('ragsum1.json')
df2=pd.read_json('lcwsum1.json')
input_header='ground_truth'
output_header=['RAG Summary','LCW Summary']
combine=pd.DataFrame({
    input_header: df1[input_header],
    output_header[0]: df1['answer'],
    output_header[1]: df2['answer']
})
combine.to_csv("Summary-Trial 1.csv", index=False)