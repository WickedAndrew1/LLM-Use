from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import SummarizationMetric, AnswerRelevancyMetric, GEval
import os
os.environ['OPENAI_API_KEY']=""
#Set os.environ to individual openAI API key.
mode="gpt-4o-mini"
sum=SummarizationMetric(
    threshold=0.1,
    n=10,
    model=mode,
    truths_extraction_limit=10
)
#Since Deepeval's summarization metric works by using the score of coverage questions and summary to assess the summary quality;
#The "n" references assessment questions, which helps evaluate the summary quality.
#It also uses extracted truths where it evaluates whether or not the response aligns with the original text. 
#The coverage score versus the alignment score.
#It then uses a minimum between these two if it passes; a score of 0 automatically fails it. This is why the threshold is incredibly low.
#In the future, establishing your own assessment questions would prove to be more accurate, since that can be inputted as a parameter.
arm = AnswerRelevancyMetric(
    threshold=0.5,
    model=mode
)
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
dataset=EvaluationDataset()
dataset.add_test_cases_from_json_file(
    file_path="lcwsum1.json",
    input_key_name="ground_truth",
    actual_output_key_name="answer",
)
y=len(dataset)
sd1=EvaluationDataset()
a=0
d=1
q=0
for i in dataset:
    if a<6:
        sd1.add_test_case(i)
    a+=1
    q+=1
    if a==6:
        id="lcwsum1-"+str(d)
        evaluate(test_cases=sd1, metrics=[sum,arm,coherency],identifier=id)
        sd1=EvaluationDataset()
        a=0
        d+=1
    e=len(sd1)
    if q==y and e!=0:
        evaluate(test_cases=sd1, metrics=[sum,arm,coherency],identifier=id)
        sd1=EvaluationDataset()

    