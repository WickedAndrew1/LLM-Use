import json
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from datasets import load_dataset

# Authentication with Hugging Face
adnan_token = ""
login(token=adnan_token)

# Initialize Llama 3.1 model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Function to generate response with token counting
def generate_response(query, max_length=512):
    # Create prompt without RAG context
    prompt = f"""<|system|>
    You are a helpful assistant who specializes in US Federal Acquisition Regulation (FAR). Answer the user's question to the best of your ability based on your knowledge of FAR.
    </|system|>

    <|user|>
    {query}
    </|user|>

    <|assistant|>
    """

    # Generate response with the model
    try:
        # Encode the prompt to get the input token count
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_token_count = len(inputs.input_ids[0])

        # Start timing
        start_time = time.time()

        # Generate the output
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # End timing
        end_time = time.time()

        # Count the generated tokens (excluding input tokens)
        generated_sequence = outputs.sequences[0]
        total_tokens = len(generated_sequence)
        generated_tokens = total_tokens - input_token_count

        # Calculate generation time and tokens per second
        generation_time = end_time - start_time
        tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0

        # Log the metrics
        print(f"Generated {generated_tokens} tokens in {generation_time:.2f} seconds")
        print(f"Generation speed: {tokens_per_second:.2f} tokens/second")

        # Decode the response
        full_response = tokenizer.decode(generated_sequence, skip_special_tokens=False)

        # Extract just what comes after "<|assistant|>" and before any closing tags
        if "<|assistant|>" in full_response:
            assistant_part = full_response.split("<|assistant|>")[-1]
            # Clean up any trailing special tokens
            for token in ["</|assistant|>", "<|endoftext|>", "</|system|>"]:
                if token in assistant_part:
                    assistant_part = assistant_part.split(token)[0]

            return assistant_part.strip(), tokens_per_second, generated_tokens, generation_time
        else:
            # Fallback
            return tokenizer.decode(outputs.sequences[0], skip_special_tokens=True), tokens_per_second, generated_tokens, generation_time
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"I encountered a problem while generating a response: {e}", 0, 0, 0

# Function to process dataset questions
def process_huggingface_dataset(dataset_name, output_file="results.json", metrics_file="metrics_summary.json"):
    try:
        # Load the dataset
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)

        if "test" in dataset:
            questions = dataset["test"]
        elif "validation" in dataset:
            questions = dataset["validation"]
        elif "train" in dataset:
            questions = dataset["train"]
        else:
            # Get the first split, whatever it is
            first_split = list(dataset.keys())[0]
            questions = dataset[first_split]

        print(f"Loaded {len(questions)} questions from dataset")

        # Prepare results container with metrics summary
        results = []
        total_tokens = 0
        total_generation_time = 0
        token_speeds = []
        per_question_metrics = []  # Store detailed metrics separately

        # Process each question
        for i, item in enumerate(questions):
            # Extract question based on dataset format (adjust as needed)
            if "question" in item:
                query = item["question"]
            elif "query" in item:
                query = item["query"]
            else:
                # Try to find a field that might contain the question
                query = None
                for key, value in item.items():
                    if isinstance(value, str) and "?" in value:
                        query = value
                        break

                if query is None:
                    print(f"Skipping item {i}: Could not identify question field")
                    continue

            print(f"\nProcessing question {i+1}/{len(questions)}: {query}")

            # Get answer with metrics directly from the model
            answer, tokens_per_second, tokens, generation_time = generate_response(query)

            # Accumulate metrics
            total_tokens += tokens
            total_generation_time += generation_time
            token_speeds.append(tokens_per_second)

            # Store detailed metrics separately
            per_question_metrics.append({
                "question_index": i,
                "question": query[:50] + "..." if len(query) > 50 else query,  # Truncated for metrics file
                "tokens_generated": tokens,
                "generation_time_seconds": generation_time,
                "tokens_per_second": tokens_per_second
            })

            # Store result for main JSON (original format)
            result = {
                "question": query,
                "answer": answer
            }

            # If the dataset has a ground truth answer, store it too
            if "answer" in item:
                result["ground_truth"] = item["answer"]

            results.append(result)

            # Print progress with metrics
            print(f"Answer: {answer[:100]}..." if len(answer) > 100 else f"Answer: {answer}")
            print(f"Generation metrics: {tokens} tokens at {tokens_per_second:.2f} tokens/sec ({generation_time:.2f} sec)")

        # Calculate overall metrics
        avg_tokens_per_second = sum(token_speeds) / len(token_speeds) if token_speeds else 0
        overall_tokens_per_second = total_tokens / total_generation_time if total_generation_time > 0 else 0

        # Create metrics summary
        metrics_summary = {
            "total_tokens_generated": total_tokens,
            "total_generation_time_seconds": total_generation_time,
            "average_tokens_per_second": avg_tokens_per_second,
            "overall_tokens_per_second": overall_tokens_per_second,
            "questions_processed": len(results),
            "per_question_metrics": per_question_metrics
        }

        # Save original format results (without metrics)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

        # Save metrics to separate file
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_summary, f, indent=4)

        print(f"\nProcessed {len(results)} questions. Results saved to {output_file}")
        print(f"Performance metrics saved to {metrics_file}")
        print(f"\nOverall performance: {total_tokens} tokens generated in {total_generation_time:.2f} seconds")
        print(f"Average generation speed: {avg_tokens_per_second:.2f} tokens/second")
        print(f"Overall generation speed: {overall_tokens_per_second:.2f} tokens/second")

        return results, metrics_summary
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return [], {"error": str(e)}

# CONFIGURATION
DATASET_NAME = "aalam24/far1-13_summary"
RESULTS_FILE = "lcwsum3.json"
METRICS_FILE = "lcwsum3_performance.json"

# Main execution
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(RESULTS_FILE) if os.path.dirname(RESULTS_FILE) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_FILE) if os.path.dirname(METRICS_FILE) else '.', exist_ok=True)

    # Process questions directly with base LLM
    print(f"\nProcessing questions from {DATASET_NAME} using base Llama 3.1 model...")
    results, metrics = process_huggingface_dataset(DATASET_NAME, RESULTS_FILE, METRICS_FILE)

    print("\nProcess complete!")
