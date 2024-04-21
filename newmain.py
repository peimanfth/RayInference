import ray
from ray.data import from_pandas
import pandas as pd
import numpy as np
# from transformers import pipeline, LlamaTokenizer, LlamaForCausalLM
from models.model_manager import ModelManager
from models.DistributedModel import DistributedModel
import time

# @ray.remote(num_cpus=5)  # Ensuring each actor uses exactly one CPU
# class DistributedModel:
#     def __init__(self, model_name, local_path=None):
#         self.model = ModelManager(model_name, local_path)
    
#     def infer(self, texts):
#         return [self.model.infer(text) for text in texts]
    
#     def batch_infer(self, texts):
#         # This method now expects a list of texts and uses the pipeline's built-in batch processing
#         return self.model.infer(texts)

def process_batch(batch, distributed_model):
    # This function now receives an actor handle and sends a list of texts for inference
    results = ray.get(distributed_model.batch_infer.remote(batch['prompt'].tolist()))
    # print(results)
    # correct_format = np.array(results, dtype=object)
    correct_format = np.array([item for sublist in results for item in sublist], dtype=object)  # Flattening

    print("----------------------This is the Bathch----------------------")
    print(batch)
    print("----------------------This is the Results----------------------")
    print(correct_format)
    print("----------------------This is the Results----------------------")

    return {'generated_text': correct_format}  # Ensure results are already 1D
    # return {'generated_text': correct_format}

# Main function
def main():
    ray.init(num_cpus=16)  # Use 4 cores
    start_time = time.time()

    local_model_paths = {
        "gpt-2": "/home/peiman/projects/RayInference/gpt-2",
        "llama-2-7b-chat-hf": "/home/peiman/projects/RayInference/llama-2-7b-chat-hf",
        "gemma-7b": "/home/peiman/projects/RayInference/gemma-7b",
        "mistral-7b-instruct": "/home/peiman/projects/RayInference/mistral-7b-instruct",
    }
    print(local_model_paths.items())

    data_path = '/home/peiman/projects/RayInference/test1.csv'
    data_df = pd.read_csv(data_path)
    data = from_pandas(data_df)

    # Map model names onto data
    data = data.map_batches(
        lambda df: df.assign(model_name=df['llm'].apply(lambda x: local_model_paths.get(x, None))),
        batch_size=10,  # Adjust batch size as needed
        batch_format='pandas'
    )
    #write batched data to a file
    data.to_pandas().to_csv('batched_data.csv')

    # Initialize DistributedModel actors
    model_actors = {model_name: DistributedModel.remote(model_name, path) 
                    for model_name, path in local_model_paths.items()}
    # print(model_actors[['gpt-2'].iloc[0]])

    # Apply the inference model to each batch using available actors
    results = data.map_batches(
        lambda batch,: process_batch(batch, model_actors[batch['llm'].iloc[0]]),
        batch_size=20,  # Specify batch size
        batch_format="pandas"
    )

    final_results = results.take(limit=100)  # Collecting results as a list of dicts
    print(len(final_results))
    end_time = time.time()

    print(final_results)
    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
