import gymnasium as gym

# Chen's shortest path frozen lake (from https://github.com/ycchen1989/Var-QuantumCircuits-DeepRL/tree/master/Code)
from gymnasium.envs.registration import register

register(
    id="Deterministic-ShortestPath-4x4-FrozenLake-v0",  # name given to this new environment
    entry_point="ShortestPathFrozenLake:ShortestPathFrozenLake",  # env entry point
    kwargs={"desc": ["SFFF", "FFFH", "FHFH", "HFFG"], "map_name": "4x4", "is_slippery": False},
    # argument passed to the env
)

# if you want to use alternate Frozen Lake env for Output Scaling Bias Test:
register(
    id="Deterministic-ShortestPath-4x4-FrozenLake-v0-alt",  # name given to this new environment
    entry_point="ShortestPathFrozenLake:ShortestPathFrozenLake",  # env entry point
    kwargs={"desc": ["FSFF", "HFFF", "GHFF", "FFFF"], "map_name": "4x4", "is_slippery": False},
)  # ["SFFF", "HHFF", "FGHF", "FFFF"]
# register(
#     id='Deterministic-4x4-FrozenLake-v0', # name given to this new environment
#     entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv', # env entry point
#     kwargs={'map_name': '4x4', 'is_slippery': False} # argument passed to the env
# )


# environment setup:
def make_env(gym_id, seed, env_num, num_envs, chkpt_dir, load_chkpt, store_envs):
    def thunk():
        if gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0":
            env = gym.make("Deterministic-ShortestPath-4x4-FrozenLake-v0")
            print("ShortestPathFrozenLake")
            env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
        elif gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt":
            env = gym.make("Deterministic-ShortestPath-4x4-FrozenLake-v0-alt")
            print("ShortestPathFrozenLake-alt-version")
            env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
        elif gym_id == "FrozenLake-v0" or gym_id == "FrozenLake-v1":
            env = gym.make(
                gym_id, desc=["SFFF", "FFFH", "FHFH", "HFFG"], map_name="4x4", is_slippery=False
            )
            print("is_slippery=False")
            env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
        else:
            env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # VectorListInfo
        # if capture_video:
        #    if env_num == 0:
        #        env = gym.wrappers.RecordVideo(
        #            env, f"videos/{run_name}", record_video_trigger=lambda t: t % 1000 == 0
        #        )

        # env.seed(seed)  #removed with v26, random env generations should be set by np.random.seed()
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        # if restarted redo all steps done in the env in the last unfinished run before restart
        if (
            load_chkpt
            and store_envs.restore_envs[env_num]
            and (
                gym_id == "FrozenLake-v0"
                or gym_id == "FrozenLake-v1"
                or gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
                or gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
            )
        ):  # does not work for environments with random starting state like cartpole
            # since we cant save the seed state at the exact time of the last env reset
            # (without makeing this extremely overcomplicated and resource intensive)
            env.reset()
            # store_envs.load_envs(chkpt_dir, num_envs) #done in main
            checkpoint_actions = store_envs.get_storage(env_num)
            if checkpoint_actions.size == 1:
                env.step(int(checkpoint_actions))
            else:
                for j in range(checkpoint_actions.size):
                    env.step(int(checkpoint_actions[j]))
            store_envs.restore_envs[env_num] = False
        return env

    return thunk
