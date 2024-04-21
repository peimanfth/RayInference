# utils/ray_utils.py
import ray

def init_ray():
    if not ray.is_initialized():
        ray.init()
