import os
os.environ['OPENAI_API_KEY']=""
#Use individual OpenAI API key.
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCaseParams

mode="gpt-4o-mini"

coherency=GEval(
    name="Coherence",
    criteria="Determine if the overall quality of sentences in 'actual output' is coherent. Focus on clarity, readability, and fluency of text.",
    evaluation_steps=[
        "To evaluate this metric, only consider the information found in actual_output. Do not deduct any points if the information stated is incorrect or does not align with the prompt. Do not deduct any points if 'actual output' incorrectly references something.",
        "To evaluate coherency, determine if the overall quality of sentences in 'actual output' is clear, fluent, and readable.",
        "Carefully read the sentences of 'actual output'",
        "Deduct significant points if 'actual output' abruptly cuts off.",
        "If the sentences in 'actual output' contain characters/symbols that are not typically found in paragraphs or sentences, significantly deduct points.",
        "If the sentences contain grammatical or spelling errors, deduct points"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=mode,
    threshold=0.6,
)
#Threshold was raised to .6 from .5. Some test cases should have failed when they didn't, simply because the threshold was too low.
accuracy=GEval(
    name="Accuracy",
    criteria="Determine if the actual output is accurate based on the expected output and input.",
    evaluation_steps=[
        "Carefully read the actual output, the expected output, and the input",
        "Given the input, compare the actual output with the expected output. When comparing them, deduct if key information is missing from 'actual output'.",
        "Determine if the 'actual output' is accurate, given the 'input' and 'expected output",
        "Do not deduct if the 'actual output' contains more in-depth information than 'expected output', while still stating correct information relating to the 'input'.",
        "Determine if the response made in 'actual output' contradicts the response made in 'expected output. If this is the case, significantly deduct points'"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=mode,
    threshold=0.4,
)
#Threshold was lowered from .5 to .4. If the actual output had correct information, but less sentences than the expected output, it'd be unfairly deducted.

relevancy= AnswerRelevancyMetric(
    threshold=0.5,
    model=mode,
)

dataset=EvaluationDataset()
dataset.add_test_cases_from_json_file(
    file_path="lcwqna3.json",
    input_key_name="question",
    actual_output_key_name="answer",
    expected_output_key_name="ground_truth"
)
y=len(dataset)
sd1=EvaluationDataset()
a=0
d=1
q=0
for i in dataset:
    if a<10:
        sd1.add_test_case(i)
    a+=1
    q+=1
    if a==10:
        id=str('lcwqna3-')+str(d)
        evaluate(test_cases=sd1,metrics=[coherency,accuracy,relevancy],identifier=id)
        sd1=EvaluationDataset()
        d+=1
        a=0
    e=len(sd1)
    if q==y and e!=0:
        evaluate(test_cases=sd1,metrics=[coherency,accuracy,relevancy],identifier=id)
        sd1=EvaluationDataset()