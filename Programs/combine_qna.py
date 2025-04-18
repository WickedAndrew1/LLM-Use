import pandas as pd
df1=pd.read_json('ragqna1.json')
df2=pd.read_json('lcwqna1.json')
input_header='question'
output_header=['RAG QNA','LCW QNA']
combine=pd.DataFrame({
    input_header: df1[input_header],
    output_header[0]: df1['answer'],
    output_header[1]: df2['answer']
})
combine.to_csv("QNA-Trial 1.csv", index=False)