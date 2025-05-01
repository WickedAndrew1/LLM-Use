import json
import os
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from datasets import load_dataset
import time

# Authentication with Hugging Face
adnan_token = ""
login(token=adnan_token)

# Initialize Llama 3.1 model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Initialize embeddings model with the correct package
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Web scraper function
def webscraper(url, output_file=None):
    try:
        # Add headers to mimic a browser to avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes
        htmlcontent = response.text
        parse = BeautifulSoup(htmlcontent, 'html.parser')

        # Extract all paragraph text
        paragraphs = [p.get_text().strip() for p in parse.find_all('p') if p.get_text().strip()]

        # Extract headers for better context
        headers = []
        for tag in ['h1', 'h2', 'h3', 'h4']:
            headers.extend([h.get_text().strip() for h in parse.find_all(tag) if h.get_text().strip()])

        # Extract list items (crucial for bulleted construction requirements)
        list_items = [li.get_text().strip() for li in parse.find_all('li') if li.get_text().strip()]

        # Combine all text
        all_text = headers + paragraphs + list_items

        # Filter out empty strings
        all_text = [text for text in all_text if text]

        if not all_text:
            print(f"Warning: No text content found on {url}")
            return None  # Return None instead of empty list

        # Save to JSON file if output_file is provided
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_text, f, indent=4)

        print(f"Scraped {len(all_text)} text elements from {url}")
        return all_text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        # Return None instead of error message to prevent processing errors as content
        return None

# Function to create Chroma vector database from the scraped content
def create_vector_db(content, persist_directory="chroma_db"):
    # Check if content is None or empty
    if content is None or not content:
        print("Cannot create vector database: No valid content available.")
        return None

    # Make sure the directory exists
    os.makedirs(persist_directory, exist_ok=True)

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # Process each item individually to avoid empty documents
    documents = []
    for text in content:
        if not text or len(text.strip()) == 0:
            continue

        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            if chunk and len(chunk.strip()) > 0:
                documents.append(Document(page_content=chunk))

    if not documents:
        # If no valid documents, return None
        print("No valid content could be processed into document chunks.")
        return None

    print(f"Created {len(documents)} document chunks for vectorization")

    # Create and persist the Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectorstore.persist()  # Make sure to persist the database

    return vectorstore

# Function to get relevant context from vector store
def get_context(query, vectorstore, k=5):
    try:
        docs = vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return None

# Modified generate_response function with token counting
def generate_response(query, context, max_length=512):
    # Create prompt with context
    prompt = f"""<|system|>
    You are a helpful assistant who specializes in US Federal Acquisition Regulation (FAR). Answer the user's question to the best of your ability based on the context. If the exact answer is not provided in the context, use pretrained knowledge to answer the questions.

    Context:
    {context}
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
            return_dict_in_generate=True,  # Return detailed output
            output_scores=True,  # Get scores
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

# Modified function to query the knowledge base with metrics
def query_knowledge_base(query, db_directory="chroma_db"):
    # Check if database directory exists
    if not os.path.exists(db_directory):
        return "Knowledge base does not exist. Please build it first.", 0, 0, 0

    # Load the vector database
    try:
        vectorstore = Chroma(
            persist_directory=db_directory,
            embedding_function=embedding_model
        )

        # Get relevant context for the query
        context = get_context(query, vectorstore, k=5)

        #if context is None:
        #    return "I couldn't find relevant information in the knowledge base to answer your question.", 0, 0, 0

        # Generate response with metrics
        response, tokens_per_second, generated_tokens, generation_time = generate_response(query, context)
        return response, tokens_per_second, generated_tokens, generation_time
    except Exception as e:
        print(f"Error querying knowledge base: {e}")
        return f"I encountered a problem while searching the knowledge base: {e}", 0, 0, 0

# NEW FUNCTION: Build knowledge base from multiple websites
def build_multi_site_knowledge_base(urls, db_directory="chroma_db"):
    all_content = []

    print(f"Building knowledge base from {len(urls)} websites...")

    for i, url in enumerate(urls):
        print(f"Processing website {i+1}/{len(urls)}: {url}")
        content = webscraper(url)

        if content:
            all_content.extend(content)
        else:
            print(f"Warning: Could not extract content from {url}")

        # Add a small delay to avoid overwhelming the server
        if i < len(urls) - 1:  # Don't sleep after the last URL
            time.sleep(2)

    if not all_content:
        print("Failed to build knowledge base: No content could be extracted from any of the provided URLs")
        return None

    print(f"Total content extracted: {len(all_content)} text elements")

    # Create vector database from all content
    vectorstore = create_vector_db(all_content, db_directory)

    if vectorstore is not None:
        print(f"Knowledge base created and stored in '{db_directory}'")
        return vectorstore
    else:
        print(f"Failed to build knowledge base: Could not create vector database")
        return None

# Modified function to process multiple questions with metrics tracking
def process_huggingface_dataset(dataset_name, db_directory="chroma_db", output_file="results.json", metrics_file="metrics_summary.json"):
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

            # Get answer with metrics
            answer, tokens_per_second, tokens, generation_time = query_knowledge_base(query, db_directory)
            
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

            # Optional: throttle requests to avoid overwhelming resources
            if i < len(questions) - 1:  # Don't sleep after the last question
                time.sleep(0.5)

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

# CONFIGURATION - Modify these values for your specific use case
WEBSITE_URLS = [
"https://www.acquisition.gov/far/part-1",
"https://www.acquisition.gov/far/part-2",
"https://www.acquisition.gov/far/part-3",
"https://www.acquisition.gov/far/part-4",
"https://www.acquisition.gov/far/part-5",
"https://www.acquisition.gov/far/part-6",
"https://www.acquisition.gov/far/part-7",
"https://www.acquisition.gov/far/part-8",
"https://www.acquisition.gov/far/part-9",
"https://www.acquisition.gov/far/part-10",
"https://www.acquisition.gov/far/part-11",
"https://www.acquisition.gov/far/part-12",
"https://www.acquisition.gov/far/part-13",
]

DATASET_NAME = "aalam24/far1-13_qa"
DB_DIRECTORY = "chroma_db"
RESULTS_FILE = "ragqa3.json"
METRICS_FILE = "ragqa3_performance.json"

# Main execution
if __name__ == "__main__":
    # Build knowledge base from multiple websites
    vectorstore = build_multi_site_knowledge_base(WEBSITE_URLS, DB_DIRECTORY)

    # Only proceed with queries if knowledge base was successfully built
    if vectorstore is not None:
        # Process questions from Hugging Face dataset
        print(f"\nProcessing questions from {DATASET_NAME}...")
        results, metrics = process_huggingface_dataset(DATASET_NAME, DB_DIRECTORY, RESULTS_FILE, METRICS_FILE)

        print("\nProcess complete!")
    else:
        print("\nCannot answer queries because knowledge base creation failed.")
        print("Please check the website URLs and ensure you have permission to access them.")
