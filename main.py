import ray
from ray.data import from_pandas
import pandas as pd
import time
from models.model_manager import ModelManager
from models.DistributedModel import DistributedModel
import gc
import torch

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def process_batch(batch, distributed_model):
    results = ray.get(distributed_model.batch_infer.remote(batch['prompt'].tolist()))
    return {'generated_text': results}

def main():
    ray.init(num_cpus=16)
    start_time = time.time()

    local_model_paths = {
        # "gpt-2": "/home/peiman/projects/RayInference/gpt-2",
        # "llama-2-7b-chat-hf": "/home/peiman/projects/RayInference/llama-2-7b-chat-hf",
        # "gemma-7b": "/home/peiman/projects/RayInference/gemma-7b",
        "gemma-2b": "/home/peiman/projects/RayInference/gemma-2b",
        # "mistral-7b-instruct": "/home/peiman/projects/RayInference/mistral-ss-instruct",
    }

    data_path = '/home/peiman/projects/RayInference/test13.csv'
    data_df = pd.read_csv(data_path)
    data = from_pandas(data_df)

    # Add model_name to the dataset
    data = data.map_batches(
        lambda df: df.assign(model_path=df['llm'].apply(lambda x: local_model_paths.get(x, None))),
        batch_size=10,
        batch_format='pandas'
    )

    # Initialize DistributedModel actors
    model_actors = {model_name: DistributedModel.remote(model_name, path)
                    for model_name, path in local_model_paths.items()}

    # Process each batch by the assigned model
    results = []
    for model_name, model_path in local_model_paths.items():
        model_data = data.filter(lambda x: x['model_path'] == model_path)
        actor = model_actors[model_name]
        model_results = model_data.map_batches(
            lambda batch: process_batch(batch, actor),
            batch_size=4,
            batch_format='pandas'
        )
        results.extend(model_results.take(limit=1000))
        # clear_memory()
        # print("--------------------------Memory Cleared--------------------------")

    end_time = time.time()
    print(f"Results obtained: {len(results)}")
    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
