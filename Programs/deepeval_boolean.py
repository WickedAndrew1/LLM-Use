import os
os.environ['OPENAI_API_KEY']=""
#Set os.environ to your own API key.
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

mode="gpt-4o-mini"

dataset=EvaluationDataset()
dataset.add_test_cases_from_json_file(
    file_path="rb3.json",
    input_key_name="question",
    actual_output_key_name="answer",
    expected_output_key_name="ground_truth"
)

y=len(dataset)
sd1=EvaluationDataset()
correct=GEval(
    name="Correctness",
    criteria="Determine if the actual output is the same as the expected output.",
    evaluation_steps=[
        "Carefully read the actual_output and the expected_output",
        "In 'actual output' find whether or not there's a 'FALSE' or a 'TRUE' in it. If there is, compare that to the response in 'expected_output'. If it is the same, pass it. If they are different, fail it",
        "If 'TRUE' or 'FALSE' is not present in 'actual_output', it's an automatic fail"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=mode,
    strict_mode=True
)
#Chain of thoughts are very important; strict mode is on, which means it has to be 1 or 0 for the threshold. 
#since we are comparing true and false, this is justifiable to have such a high threshold.
#Chain of thoughts have to be steps following each other, for it to properly work.

prompt=GEval(
    name="Prompt Alignment",
    criteria="Determine if the actual output only has 'TRUE' or 'FALSE' in its response.",
    evaluation_steps=[
        "Carefully read the actual output",
        "If the actual output contains any other words than 'TRUE' or 'FALSE' in its response, fail it automatically.",
        "If the actual output does not contain the words 'TRUE' or 'FALSE' in its response, fail it automatically.",
        "If the actual output only contains 'TRUE' or 'FALSE', it automatically passes"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=mode,
    strict_mode=True
)
a=0
d=1
q=0

for i in dataset:
    if a<20:
        sd1.add_test_case(i)
    a+=1
    q+=1
    if a==20:
        id='rb3-'+ str(d)
        evaluate(test_cases=sd1,metrics=[correct, prompt],identifier=id)
        sd1=EvaluationDataset()
        a=0
        d+=1
    e=len(sd1)
    if q==y and e!=0:
        evaluate(test_cases=sd1,metrics=[correct, prompt],identifier=id)
        sd1=EvaluationDataset()