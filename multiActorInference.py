import ray
import pandas as pd
from ray.data import from_pandas
import time
import datetime
import os
import logging
import psutil  # To monitor memory usage
import json  # For storing results in a structured format

from models.model_manager import ModelManager
from models.DistributedModel import DistributedModel
import gc
import torch
import random

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()



@ray.remote(num_cpus=1)
def process_batch(batch, distributed_model, batch_size, model_name, batch_number):
    start_time = time.time()
    # distributed_model.add_pending_task.remote(batch['prompt'].tolist())
    # initial_mem = psutil.Process(os.getpid()).memory_info().rss  # Memory usage before processing
    results = ray.get(distributed_model.batch_infer.remote(batch['prompt'].tolist()))
    distributed_model.remove_pending_task.remote(batch['prompt'].tolist())
    distributed_model.update_task_count.remote(True)

    model_memory_usage = ray.get(distributed_model.get_model_memory_usage.remote())


    end_time = time.time()

    # final_mem = psutil.Process(os.getpid()).memory_info().rss  # Memory usage after processing
    # memory_used = final_mem - initial_mem  # Memory occupied during processing

    inference_time = end_time - start_time
    resultDict = {}
    for idx,result in enumerate(results):
        resultDict[idx] = result[0]['generated_text']
        
    # Batch-level information now includes batch number
    batch_info = {
        'model_type': model_name,
        'batch_size': batch_size,
        'inference_time': inference_time,
        'memory_used': model_memory_usage,
        'batch_number': batch_number  # Track which batch number this is within the current model and batch size context
    }
    return {
        'generated_text': [resultDict],
        'batch_info': [batch_info]  # Encapsulate batch-level info
    }

def get_least_loaded_actor(actors):
    """Select the actor with the minimum queued tasks."""
    min_tasks = float('inf')
    selected_actor = None
    for actor in actors:
        pending_tasks = len(ray.get(actor.get_pending_tasks.remote()))
        logging.info(f"Actor {actor} has {pending_tasks} pending tasks.")
        if pending_tasks < min_tasks:
            min_tasks = pending_tasks
            selected_actor = actor
    return selected_actor

def weighted_actor_selection(actors, weights):
    """Select an actor randomly, weighted by the given weights."""
    total = sum(weights)
    cumulative_weights = [sum(weights[:i+1]) / total for i in range(len(weights))]
    r = random.random()
    for actor, cum_weight in zip(actors, cumulative_weights):
        if r < cum_weight:
            return actor
    return actors[-1]

def random_actor_selection(actors):
    """Select an actor randomly."""
    return random.choice(actors)

def select_actor_by_hash_modulo(batch, actors):
    """Select an actor based on a hash of the batch content."""
    batch_hash = hash(tuple(batch))  # Example: hash the batch data to get a single integer
    index = batch_hash % len(actors)
    return actors[index]

def select_actor_by_hash_modulo_index(idx, actors):
    """Select an actor based on a hash of the batch content."""
    idxHash = hash(idx)  # Example: hash the batch data to get a single integer
    index = idxHash % len(actors)
    return actors[index]
def RoundRobinSelection(idx,actors):
    return actors[idx % len(actors)]

def RandomSelection(actors):
    return random.choice(actors)
def HashBasedSelection(batch,actors):
    prompts = batch['prompt'].tolist()
    hashPrompts = [hash(prompt) for prompt in prompts]
    logging.info(f"Hashed prompts: {hashPrompts}")
    #average hash
    index = sum(hashPrompts) // len(hashPrompts) % len(actors)
    return actors[index]



def run_experiment(data, batch_sizes, num_actors, local_model_paths):
    all_results = []
    for batch_size in batch_sizes:
        DistributedModel.options(num_gpus=1/num_actors).remote

        model_actors = {model_name: [DistributedModel.options(num_gpus=1/num_actors).remote(model_name, path, batch_size) for _ in range(num_actors)]
                            for model_name, path in local_model_paths.items()}
        start_time = time.time()
        futures = []
        for model_name, actors in model_actors.items():
            model_data = data.filter(lambda x: x['model_path'] == local_model_paths[model_name])
            batch_number = 0
            for i, batch in enumerate(model_data.iter_batches(batch_size=batch_size, batch_format='pandas')):
                
                # actor = RoundRobinSelection(batch_number,actors) # Round-robin actor selection
                # actor = RandomSelection(actors) # Random actor selection
                actor = HashBasedSelection(batch,actors) # Hash-based actor selection



                logging.info(f"all actors: {actors}")      
                logging.info(f"Processing batch {batch_number} for model {model_name} with batch size {batch_size} using {actor} with index {actors.index(actor)}")
                actor.add_pending_task.remote(batch['prompt'].tolist())
                # weights = [1.0 / (1 + ray.get(actor.get_task_count.remote())) for actor in actors] # Weighted selection based on pending tasks
                futures.append(process_batch.remote(batch, actor, batch_size, model_name, batch_number))
                # actor = weighted_actor_selection(actors, weights)
                batch_number += 1

        # Collect results as they complete
        results = []
        while futures:
            done, futures = ray.wait(futures)
            result = ray.get(done[0])
            # for actorr in actors:
            #     logging.info(f"Actor {actorr} has task count of {ray.get(actorr.get_task_count.remote())} and {len(ray.get(actorr.get_pending_tasks.remote()))} pending tasks.")
            results.append(result)

        end_time = time.time()
        results_summary = {
            'batch_size': batch_size,
            'total_time': end_time - start_time,
            'results': results
        }
        all_results.append(results_summary)
    return all_results


 
def main():
    ray.init(num_cpus=16, num_gpus=1)
    local_model_paths = {
        "gpt-2": "/home/peiman/projects/RayInference/gpt-2",
        # "gemma-2b": "/home/peiman/projects/RayInference/gemma-2b",
    }

    data_path = '/home/peiman/projects/RayInference/unique_labeled_prompts.csv'
    data_df = pd.read_csv(data_path)
    data = from_pandas(data_df)

    data = data.map_batches(
        lambda df: df.assign(model_path=df['llm'].apply(lambda x: local_model_paths.get(x))),
        batch_size=10,
        batch_format='pandas'
    )

    max_actors = 3

    for num_actors in range(3, max_actors + 1):

        # DistributedModel.options(num_gpus=1/num_actors).remote

        # model_actors = {model_name: [DistributedModel.options(num_gpus=1/num_actors).remote(model_name, path) for _ in range(num_actors)]
        #                 for model_name, path in local_model_paths.items()}

        batch_sizes = [2**i for i in range(2, 3)]  # Batch sizes 16, 32, 64, 128, 256

        experiment_results = run_experiment(data, batch_sizes, num_actors, local_model_paths)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f'logs/multiActor/schedulers/experiment_results_{num_actors}_{current_time}.json'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as f:
            json.dump({data_path: experiment_results}, f, indent=4)

        print(f'Experiment {num_actors} actors completed and results saved."')
    print("All experiments completed.")
    ray.shutdown()





if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(filename="ray_application.log", level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
