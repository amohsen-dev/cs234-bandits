import numpy as np
from utils import process_data
from learners import LinUCBLearner, NeuralLinUCBLearner, LinearThompsonSamplingLearner


if __name__=='__main__':
    data = process_data()
    np.random.seed(234)
    seeds = np.random.randint(1, 999, 20)
    LEARNERS = {}
    LEARNERS['UCB'] = {seed: LinUCBLearner(data, seed=seed) for seed in seeds}
    LEARNERS['Neural'] = {seed: NeuralLinUCBLearner(data, seed=seed) for seed in seeds}
    LEARNERS['Thompson'] = {seed: LinearThompsonSamplingLearner(data, seed=seed) for seed in seeds}
    for k, d in LEARNERS.items():
        for seed, lrn in d.items():
            lrn.create_agent()
            r = lrn.learn()
            with open(f'results/{k}_{seed}.npy', 'wb') as f:
                np.save(f, r)