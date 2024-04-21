from models.model_manager import ModelManager
import ray
import random
import time
import os
import subprocess


@ray.remote(num_gpus=1/6)  # Ensuring each actor uses exactly one CPU
class DistributedModel:
    def __init__(self, model_name, local_path=None, batch_size=None):
        self.model = ModelManager(model_name, local_path, batch_size)
        self.pending_tasks = []
        self.task_count = 0

    def update_task_count(self, increment):
        if increment:
            self.task_count += 1
        else:
            self.task_count -= 1
    def add_pending_task(self, texts):
        self.pending_tasks.append(texts)

    def remove_pending_task(self, texts):
        self.pending_tasks.remove(texts)

    def get_task_count(self):
        return self.task_count
    
    def get_pending_tasks(self):
        return self.pending_tasks
    
    def infer(self, texts):
        return [self.model.infer(text) for text in texts]
    
    def batch_infer(self, texts):
        # This method now expects a list of texts and uses the pipeline's built-in batch processing
        # self.pending_tasks.append(texts)
        # time.sleep(random.randint(1,10))
        result = self.model.infer(texts)
        # self.update_task_count(True)
        #randomly wait for 1-10 seconds
        # time.sleep(random.randint(1,10))
        # self.pending_tasks.remove(texts)
        # update_future = self.update_task_count.remote(False)
        # ray.get(update_future)
        return result
    # def get_model_memory_usage(self):
    #     """ Get the memory usage of the model. """
    #     return self.model.calculate_model_size2()
    def get_model_memory_usage(self):
        """ Get the GPU memory usage of the model. """
        pid = os.getpid()
        result = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'])
        processes = result.decode('utf-8').strip().split('\n')
        for process in processes:
            process_pid, memory = process.split(', ')
            if int(process_pid) == pid:
                return int(memory.split(' ')[0])  # Return memory usage in MiB
        return 
