import ray
import pandas as pd
from ray.data import from_pandas
import time
import datetime
import os
import psutil  # To monitor memory usage
import json  # For storing results in a structured format

from models.model_manager import ModelManager
from models.DistributedModel import DistributedModel
import gc
import torch

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# def process_batch(batch, distributed_model, batch_size):
#     start_time = time.time()
#     initial_mem = psutil.Process(os.getpid()).memory_info().rss
#     try:
#         results = ray.get(distributed_model.batch_infer.remote(batch['prompt'].tolist()))
#         # Ensure results are in a consistent format, e.g., convert complex types to string if needed
#         results = [str(result) for result in results]
#     except Exception as e:
#         print(f"Error processing batch: {e}")
#         results = ["error"] * len(batch['prompt'])  # Placeholder to maintain array length consistency
#     end_time = time.time()

#     final_mem = psutil.Process(os.getpid()).memory_info().rss
#     memory_used = final_mem - initial_mem

#     inference_time = end_time - start_time
#     return {
#         'generated_text': results,
#         'batch_size': [batch_size] * len(results),  # Ensure consistent length
#         'inference_time': [inference_time] * len(results),
#         'memory_used': [memory_used] * len(results)
#     }
def process_batch(batch, distributed_model, batch_size, model_name, batch_number):
    start_time = time.time()
    initial_mem = psutil.Process(os.getpid()).memory_info().rss  # Memory usage before processing
    results = ray.get(distributed_model.batch_infer.remote(batch['prompt'].tolist()))


    end_time = time.time()

    final_mem = psutil.Process(os.getpid()).memory_info().rss  # Memory usage after processing
    memory_used = final_mem - initial_mem  # Memory occupied during processing

    inference_time = end_time - start_time
    resultDict = {}
    for idx,result in enumerate(results):
        resultDict[idx] = result[0]['generated_text']
        
    # Batch-level information now includes batch number
    batch_info = {
        'model_type': model_name,
        'batch_size': batch_size,
        'inference_time': inference_time,
        'memory_used': memory_used,
        'batch_number': batch_number  # Track which batch number this is within the current model and batch size context
    }
    return {
        'generated_text': [resultDict],
        'batch_info': [batch_info]  # Encapsulate batch-level info
    }


# def run_experiment(data, batch_sizes, model_actors, local_model_paths):
#     all_results = []
#     for batch_size in batch_sizes:
#         start_time = time.time()
#         results = []
#         for model_name, model_path in local_model_paths.items():
#             model_data = data.filter(lambda x: x['model_path'] == model_path)
#             actor = model_actors[model_name]
#             model_results = model_data.map_batches(
#                 lambda batch: process_batch(batch, actor, batch_size),
#                 batch_size=batch_size,
#                 batch_format='pandas'
#             )
#             batch_results = model_results.take(limit=1000)
#             results.extend(batch_results)
#             clear_memory()
#         end_time = time.time()
#         all_results.append({
#             'batch_size': batch_size,
#             'total_time': end_time - start_time,
#             'results': results
#         })
#     return all_results

def run_experiment(data, batch_sizes, model_actors, local_model_paths):
    all_results = []
    for batch_size in batch_sizes:
        start_time = time.time()
        results = []
        for model_name, model_path in local_model_paths.items():
            model_data = data.filter(lambda x: x['model_path'] == model_path)
            actor = model_actors[model_name]
            batch_number = 0  # Initialize batch number for each model and batch size combination
            model_results = model_data.map_batches(
                lambda batch: process_batch(batch, actor, batch_size, model_name, batch_number),
                batch_size=batch_size,
                batch_format='pandas'
            )
            batch_results = model_results.take(limit=1000)
            results.extend(batch_results)
            batch_number += 1  # Increment batch number for each batch processed
            # clear_memory()
            end_time = time.time()
        all_results.append({
            'batch_size': batch_size,
            'total_time': end_time - start_time,
            'results': results
        })
    return all_results


def main():
    ray.init(num_cpus=16)
    local_model_paths = {
        # "gemma-7b": "/home/peiman/projects/RayInference/gemma-7b",
        "gpt-2": "/home/peiman/projects/RayInference/gpt-2",
        # "gemma-2b": "/home/peiman/projects/RayInference/gemma-2b",
        # "llama-2-7b-chat-hf": "/home/peiman/projects/RayInference/llama-2-7b-chat-hf",
        # "mistral-7b-instruct": "/home/peiman/projects/RayInference/mistral-7b-instruct"
    }

    data_path = '/home/peiman/projects/RayInference/unique_labeled_prompts.csv'
    data_df = pd.read_csv(data_path)
    data = from_pandas(data_df)

    data = data.map_batches(
        lambda df: df.assign(model_path=df['llm'].apply(lambda x: local_model_paths.get(x))),
        batch_size=10,
        batch_format='pandas'
    )

    model_actors = {model_name: DistributedModel.remote(model_name, path)
                    for model_name, path in local_model_paths.items()}

    # Define batch sizes to experiment with
    batch_sizes = [2**i for i in range(4, 9)]  # Batch sizes 1, 2, 4, 8

    experiment_results = run_experiment(data, batch_sizes, model_actors, local_model_paths)

    # Store results in a JSON file for later analysis
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create the filename with the current time
    filename = f'logs/experiment_results_{current_time}.json'

    # Create the logs directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        json.dump({data_path: experiment_results}, f, indent=4)

    print("Experiment completed and results saved.")

if __name__ == "__main__":
    main()
