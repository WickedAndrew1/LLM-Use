#Due to overlap of installations, refer to the LLM-Bert.py documentation first.
#Run command "pip install -U deepeval" 
#No need to login for deepeval. All evaluations will be done locally; may be done online as well, to monitor some data due to hopper cluster being used.
# On your command terminal, run this program.
#Steps were taken from: https://docs.confident-ai.com/docs/getting-started and https://docs.confident-ai.com/docs/integrations-huggingface
import os
from huggingface_hub import login
os.environ['HUGGINGFACEHUB_API_TOKEN']='"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'
#Finegrained-login(token="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")
login(token="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") #Read Token.
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
#Importing OS with specific use of token and environment allowed deepeval to properly register the API keys.
#Did not involve adding to PATH.
#Before running this, key note: since I am using the base instruct model, it will take a long time to run;
#It will also require a lot of CPU and Memory. If necessary, use a quantized model for your specifications.

class llama(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        device = "cpu" # the device to load the model onto-will be different for hopper cluster.

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

llama32= llama(model=model, tokenizer=tokenizer)
print(llama32.generate("Write me a joke"))
#Code was all taken from Deep-Eval's implementation.
