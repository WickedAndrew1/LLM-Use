#When installing python, select it to be on the PATH. It also must be version 3.9-version 3.12. Current Python version is 3.12.6.
#I ran into compability issues with running torch to install pytorch. It's only supported on those versions.
#Downloading python as 64-bit should also prevent any issues as well. After attempting to download bert_score,
#I noticed that pip, torch, and bert_score would never properly download. This is probably due to a lack of support
#with python's newest version(3.13.0). 
#Download C++ from Visual Studio Build tools.
#All further installs should be done within virtual environment.
#"pip install evaluate" was used to install evaluate metrics found on hugging face.
#"pip install transformers": this needed both rust and cargo packages on PATH. Download from here https://rustup.rs/ 
#If running into rustup installation issues, try to ensure it's fully downloaded, then run "rustup default stable". That should fix most issues.
#"pip install torch" to install pytorch. Key note-pytorch was installed without CUDA and with pip package due to current OS. On hopper cluster, most likely can be downloaded with CUDA.
#"pip install bert_score" for use of BERT scoring metric. It'll store a BERT-scorer that works with your machine and script packages.
#Note, testing bert_scoring still requires significant memory. 
#Errors/warnings do occur asking about caching with symlinks enabled or not. Running python as an administrator should fix this.
#States that training this model may be necessary for predictions and inference, which shouldn't be the case.
#At least for this analysis, it should be sufficient to use.
#Current BERT is Roberta Large Model- 'roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.46.2)'
#Use of command: huggingface-cli scan-cache -v should allow size on disk and what else to be shown.
#For use, references serve as the original text and predictions is what's being used to compare. 
#For summarization metric, we can't use it to compare to the original text due to difference in sentences.
#It should involve a summary from a dataset or a human-generated one and comparing with bert-scoring for LLM summary.
from evaluate import load
bertscore = load("bertscore")
predictions = ["hello there, it is I", "general kenobi is here"]
references = ["hello there, this is charles xavier", "kenobi is dead"]
results = bertscore.compute(predictions=predictions, references=references, lang="en")
print(results)
#Key notes-this is without pipelines, transformers, and specific bert variables. This is simply done to ensure Bert-scoring works.
