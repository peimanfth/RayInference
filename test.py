from multiActorInference import select_actor_by_hash_modulo_index


if __name__ == "__main__":
    actors = [1, 2, 3, 4, 5]
    idx = "ASDASf"
    selected_actor = select_actor_by_hash_modulo_index(idx, actors)
    print(selected_actor)