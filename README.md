# information-theoretic-bandit
This repository currently implements the testbed for multi-arm bandit problem presented in Chapter 2 of [1].
## TODO:
Currently, the testbed outputs results only for single experiment run. Should be extended to allow experiments and visualizations for multiple runs.

## To run
To run a single bandit experiment:
```
usage: testbed.py [-h] [-v VERBOSE] [--num_arms NUM_ARMS]
                  [--arms_mean ARMS_MEAN]
                  [--arms_mean_params [ARMS_MEAN_PARAMS [ARMS_MEAN_PARAMS ...]]]
                  [--arms_std ARMS_STD]
                  [--arms_std_params [ARMS_STD_PARAMS [ARMS_STD_PARAMS ...]]]
                  [--agent_class AGENT_CLASS] [--num-episodes NUM_EPISODES]
                  [--result-file RESULT_FILE]

K-armed bandit testbed

optional arguments:
  -h, --help            show this help message and exit
  -v VERBOSE, --verbose VERBOSE
                        Verbose
  --num_arms NUM_ARMS, -k NUM_ARMS
                        Number arms for the bandit
  --arms_mean ARMS_MEAN
                        Distribution to select mean for each arm from. Should
                        correspond to a distribution in numpy.random module.
                        Use --arms_mean_params to change the parameters passed
                        to the distribution. For example: --arms_mean=constant
                        --arms_mean_params=1 will result in bandit whose all
                        arms have mean of 1, and --arms_mean_params=normal
                        --arms_mean_params=[0,1] will result in bandit whose
                        all arms have mean drawn from numpy.random.normal(0,
                        1)
  --arms_mean_params [ARMS_MEAN_PARAMS [ARMS_MEAN_PARAMS ...]]
                        params to be passed to the distribution
                        functionspecified by --arms_mean argument.
  --arms_std ARMS_STD   Distribution to select standard deviation for each arm
                        from. See help for --arms_mean for more information
  --arms_std_params [ARMS_STD_PARAMS [ARMS_STD_PARAMS ...]]
                        params to be passed to the distribution
                        functionspecified by --arms_mean argument.
  --agent_class AGENT_CLASS, --agent_cls AGENT_CLASS
                        Name of the class (defined in agents.py) to be used as
                        an agent. Defaults to agents.DefaultAgent.
  --num-episodes NUM_EPISODES
                        Number of episodes
  --result-file RESULT_FILE
                        File to write results to
```

For example, running `10-armed bandit`, where arm value function means are taken from normal distribution with mean `μ=0` and standard deviation `σ=1.0`, and writing results in `results/result-1.json`:
```
python ./testbed.py --num_arms=5 --arms_mean="normal" --arms_mean_params 0 1 --result-file="./results/result-1.json"
```
## To visualize the results
The results recorded by `testbed.py` can be visualized with `vis.py`:
```
usage: vis.py [-h] --result-file RESULT_FILE

Data visualization for K-armed bandit.

optional arguments:
  -h, --help            show this help message and exit
  --result-file RESULT_FILE
                        File to read results from
```
For example, to visualize results from `./results/result-1.json`:
```
python ./vis.py --result-file=./results/result-1.json
```


[1] Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction (Vol. 1, No. 1). Cambridge: MIT press.

