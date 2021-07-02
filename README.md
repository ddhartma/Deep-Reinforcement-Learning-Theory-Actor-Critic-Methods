[image1]: assets/1.png
[image2]: assets/2.png
[image3]: assets/3.png
[image4]: assets/4.png
[image5]: assets/5.png
[image6]: assets/6.png
[image7]: assets/7.png
[image8]: assets/8.png
[image9]: assets/9.png
[image10]: assets/10.png
[image11]: assets/11.png
[image12]: assets/12.png
[image13]: assets/13.png
[image14]: assets/14.png
[image15]: assets/15.png
[image16]: assets/16.png
[image17]: assets/17.png
[image18]: assets/18.png


# Deep Reinforcement Learning Theory - Actor-Critic Methods

## Content
- [Introduction](#intro)
- [Motivation](#mot)
- [Bias and Variance](#bias)
- [Two Ways For Estimating Expected Returns](#exp_ret)
- [Baselines and Critics](#baselines)
- [Policy-based, Value-based and Actor-Critic](#actor_critic)
- [A Basis Actor-Critic Agent](#basis)
- [A3C: Asynchronous Advantage Actor-Critic, N-step Bootstrapping](#a3c_1)
- [A3C: Asynchronous Advantage Actor-Critic, Parallel Training](#a3c_2)
- [A3C: Asynchronous Advantage Actor-Critic, Off-policy vs. On-policy](#a3c_3)
- [A2C: Advantage Actor-Critic](#a2c)
- [A2C: Code Walk-through](#a2c_code)
- [GAE: Generalized Advantage Estimation](#gae)
- [DDPG: Deep Deterministic Policy Gradient, Continuous Action-space](#ddpg_1)
- [DDPG: Deep Deterministic Policy Gradient, Soft Updates](#ddpg_2)
- [DDPG: Code Walk-through](#ddpg_code)
- [DDPG: Code Implementation](#ddpg_impl)
- [DDPG: Paper Walk-through](#ddpg_paper)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Introduction <a id="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

### Overview:
- ***Policy-Based Methods***: methods such as
    - hill climbing
    - simulated annealing
    - adaptive noise scaling
    - cross-entropy methods
    - evolution strategies

- ***Policy Gradient Methods***:
    - REINFORCE
    - lower the variance of policy gradient algorithms

- ***Proximal Policy Optimization***:
    - Proximal Policy Optimization (PPO), a cutting-edge policy gradient method

- ***Actor-Critic Methods***
    - how to combine value-based and policy-based methods
    - bringing together the best of both worlds, to solve challenging reinforcement learning problems


## Motivation <a id="mot"></a>
- Methods that **learn both policies and value functions** are referred to as **actor-critic** because the **policy, which selects actions,** can be seen as an **actor**, and the **value function, which evaluates policies,** can be seen as a **critic**.
- Actor-critic methods often **perform better than value-based or policy-gradient methods** alone on many of the deep reinforcement learning benchmarks.

    ![image1]

### Actor-critic methods:
- Use a value function as a baseline
- Train a **neural network to approximate a value function** and then **use it as a baseline**
- All actor-critic methods use **value-based techniques** to further **reduce the variance of policy-based methods**.

## Bias and Variance <a id="bias"></a>
- What is Bias and Variance?

    ![image2]

- There is a bias-variance tradeoff, when an agent tries to estimate value functions or policies from returns.
- A **return** is calculated using **a single trajectory**.
- **Value functions** are calculated using the **expectation of returns**.
- ***Aim of coding***: **Reduce the variance** of algorithms while keeping **bias to a minimum**.

    ![image3]

- ***Goal of agent***: find policies to **maximize the total expected reward**.
- ***Problem***: But since we're **limited to sampling** the environment,
**we can only estimate these expectation**. This can lead to high bias and variance
- The question is, what's the best way to
estimate value functions.

## Two Ways For Estimating Expected Returns <a id="exp_ret"></a>
- ***Monte Carlo Estimate***:
    - The Monte-Carlo estimate consists of **rolling out an episode** and calculating the discounted total reward from the rewards sequence.
    - For example, in an episode A, you start in state **S<sub>t</sub>**, take action **A<sub>t</sub>** and a reward **R<sub>t+1</sub>** and sends you to a new state **S<sub>t+1</sub>**, ... and so on until you reach the end of the episode.
    - The Monte-Carlo estimate just add other rewards up, whether discounted or not.
    - When you then have a collection of episodes A, B, C, and D, some of those episodes will have trajectories that go through the same states.
    - Each of these episodes can give you **a different Monte-Carlo estimate G<sub>t</sub>** for the **same value function**.
    - **Calculate the value function** via an **average of the estimates**.
    - Obviously, **the more estimates** you have when taking the average, **the better the value function** will be.

- ***Temporal Difference (TD) Estimate***:
    - Say we're estimating a state value function V.
    - For estimating the **value of the current state**, it uses a **single rewards sample**.
    - Estimation of the discounted total return starts from the next state onwards.
    - You're **estimating with an estimate**.
    - For example, in episode A, you start in state **S<sub>t</sub>**, take action **A<sub>t</sub>**, get a reward **R<sub>t+1</sub>**, and sends you to a new state **S<sub>t+1</sub>**.
    - But then you can actually stop there.
    - Use **bootstrapping**, which basically means that you can use the current estimate for the next state in order to calculate a new estimate for the value function of the current state.
    - Now, the estimates of the next state will probably be off particularly early on, but that value will become better and better as your agent sees more data.
    - After doing this many many times, you will have estimated the desired value function well.

    ![image4]

- ***Monte-Carlo estimates*** will have **high variance** because
estimates for a state can vary greatly across episodes. But Monte-Carlo methods are **unbiased**. You are **not estimating using estimates**. You are only using the **true rewards** you obtained. So, given lots and lots of data, your estimates will be accurate.

    ![image5]

- ***TD estimates*** have **low variance** because you're only compounding a single time step of randomness instead of a full rollout. Though because you're **bootstrapping** on
the next state estimates and those are **not true values**,
you're **adding bias** into your calculations. Your agent will learn faster, but we'll have more problems converging.

    ![image6]


## Baselines and Critics <a id="baselines"></a>
- Check this out: [Understanding Baseline Techniques for REINFORCE](https://medium.com/@fork.tree.ai/understanding-baseline-techniques-for-reinforce-53a1e2279b57)
- The return **G** is calculated from the total discounted return. This is simply a Monte Carlo estimate (with high variance). You use then a **baseline to reduce the variance** of the REINFORCE algorithm.   
- The **baseline can be learned by using DEEP LEARNING**.
- Monte Carlo estimate has high variance and no bias.
- TD estimate has low variance and low bias.
- ***Critic implies that bias has been introduced***
- You can use Monte Carlo or TD estimates to train baselines.
- By using **MC estimates** to train baselines we have **no CRITIC**.
- By using **TD estimates** to train baselines we have a **CRITIC**.
- With **TD estimates** we are **introducing bias** but also we are **reducing variance** thus **improving convergence properties** and **speeding up learning**.
- ***Goal of In Actor-Critic Methods***: Try to continuously reduce the high-variance associated with policy-based agents. Use TD estimates to achieve this goal.
- Actor-Critic methods show faster learning than policy-based agents alone and better convergence than value-based agents alone.

## Policy-based, Value-based and Actor-Critic <a id="actor_critic"></a>
- In Policy-based methods the agent is learning to act (like playing a game)
- In Value-based methods the agent is learning to estimate situations and actions
- **GOOD IDEA**: Combine these two approaches
- In ***pure Policy-based methods*** you need lots of data and lots of time for training (to increase the probability of actions that lead to a win and decrease the probability of actions that lead to losses) --> ***inefficient approach***
- In addition: Many actions within a game that ended up in a loss could have been really good actions. --> ***Decreasing the probability of good actions taken in a lost game is not optimal***. Using this information could speed up learning further.
- A ***Critic or Value-based approach*** learns differently: At each time step the agent guesses what the final score is going to be. Guesses become better and better. The agent can separate good from bad situations better and better. The better the agent can make these distinctions the better the agent performs. Guesses introduce a bias (especially in the beginning, as guesses are wrong without experience). Guesses are prone to under or overestimation.

    ![image7]

### Why Actor-Critic Agent?
- ***Actor-Critic Agents*** learn by playing games and **adjusting probabilities of good and bad action sequences** (**policy-based** agent) and use a **critic to distinguish good from bad actions** during the game ***to speed up learning and to enhance convergence*** (**value-based** agent).

## A Basis Actor-Critic Agent <a id="basis"></a>
- An Actor-Critic Agent uses **Function Approximation** to learn a **policy** and a **value function**
- **Two neural networks**:
    - **actor** network: Takes in **a state** and outputs a policy (which defines a **distribution over actions conditioned on states**, π(a|s) or learn the parameters **θ** of this functional approximation.)
    - **critic** network: Takes in **a state** and outputs a **state value function of policy π**, **V<sub>π</sub>**
- The **critic** will learn to evaluate the **state value function** **V<sub>π</sub>** using the **TD estimate**.
- Using the critic we will calculate the advantage function and train the actor using this value.

### Input - Output flow:
- State **s** as input
- Get experience tuple **(s, a, r, s')**
- Use the TD estimate to train the critic
- Use the critic to calculate the Advantage function
- Train the actor using the calculated Advantage as a baseline

    ![image8]

## A3C: Asynchronous Advantage Actor-Critic, N-step Bootstrapping <a id="a3c_1"></a>
- Actor-Critic Agent as before
- Instead of TD estimate agent uses ***n-step bootstrapping***: A generalization of TD and Monte-Carlo estimates
- TD is a one-step bootstrapping. Agent experiences one-time-step of real rewards and bootstraps right there
- Monte-Carlo does not really bootstrap or let's say it is an infinite step bootstrapping
- **n-step bootstrapping** takes n-steps before bootstrapping.

    ![image9]

- n-step bootstrapping means that the agent will wait a little bit longer, i.e. ***elongate exploration***, before it guesses what the final score will look like (i.e. calculates the expected return of the original state).
- Benefit: ***Faster convergence*** with ***less experience required*** and ***reduction of bias***.
- In Practice: 4 or 5 steps bootstrapping are often the best.


## A3C: Asynchronous Advantage Actor-Critic, Parallel Training <a id="a3c_2"></a>
 - Unlike DQN, AC3 does not use a replay buffer.
 - In DQN we needed a replay buffer to decorrelate state-action tuples of consecutive time steps (experience at time step t+1 will be correlated to experience at time step t)

    ![image10]

### Parallel training
- A3C **replaces** the replay buffer with **parallel training**.
- Create multiple instances of the environment and the agent
- Run them all at the same time
- Agent will receive minibatches of experiences just as we need.
- Samples will be decorrelated because agents will likely experience different states at any given time.
- Besides on-policy learning (more stable learning) is possible.

    ![image11]


## A3C: Asynchronous Advantage Actor-Critic, Off-policy vs. On-policy <a id="a3c_3"></a>
- Helpful medium blog post on policies [Off-policy vs On-Policy vs Offline Reinforcement Learning Demystified!](https://kowshikchilamkurthy.medium.com/off-policy-vs-on-policy-vs-offline-reinforcement-learning-demystified-f7f87e275b48)
- Check out my GitHub page on [Temporal Difference Methods - SARSA and Q-Learning](https://github.com/ddhartma/Deep-Reinforcement-Learning-Theory-Temporal-Difference-Methods#sarsamax). Check out the images and tables on SARSA and Q-Learning to understand the on and off policy difference.
- ***On-Policy learning***: Policy which is used for interacting with the environment is also the policy being learned (e.g. SARSA)
- ***Off-Policy learning***: Policy which is used for interacting with the environment is different from the policy being learned (Q-Learning)
- In SARSA (On-Policy learning): The action for calculating the TD target and TD error is the action of the following time step **A'**.
- In Q-Learning (Off-Policy learning): The action used for calculating the TD target is the action with the highest value. Here this is not necessarily **A'**.
- In Q-Learning the agent may choose an exploratory action in the next step. Q-learning learns a deterministic optimal policy.
- In SARSA that action (exploratory or not) is already been chosen. SARSA learns the best exploratory policy.

    ![image12]

- DQN ia also an off-policy method. Agent behaves with some exploratory greedy policy to learn the optimal policy.

- When using off-policy learning agents are able to learn from many different sources including experiences generated by all versions of the agent itself (~the replay buffer).
- Problem of off-policy learning: known to be unstable, diverge with neural networks.

### What kind of policy uses A3C now?
- A3C is an ***on-policy learning*** method.

- [Q-Prop paper](https://arxiv.org/abs/1611.02247)

## A2C: Advantage Actor-Critic <a id="a2c"></a>
### What means ***Asynchronous*** in 'Asynchronous Advantage Actor-Critic'?
- In A3C: Each agent uses a **local copy of the network** to collect experience, calculate and **accumulate gradient updates** across **multiple time steps** and applies them **asynchronously** to a neural network.
- Asynchronous means that each agent will update the network on its own.
- They weights which one agent is using might be different from the other agents' weights at any given time.

### Synchoronous version of A3C: Advantage Actor-Critic --> A2C
A2C **synchronizes** all agents:
- It **waits for all agents** to finish a segment of interaction with its copy of the environment.
- Then it **updates the network at once**.
- Then updated weights will be sent back to all agents.

Consider:
- A2C is simpler to implement
- Gives pretty much the same results as A3C
- A3C is most easily trained on CPU
- A2C is normally trained on GPU

![image13]


## A2C: Code Walk-through <a id="a2c_code"></a>
- GitHub repo of [ShangtongZhang - Deep RL](https://github.com/ShangtongZhang/DeepRL)
- Video tutorial on YouTube [Video](https://www.youtube.com/watch?time_continue=7&v=LiUBJje2N0c&feature=emb_logo)


## GAE: Generalized Advantage Estimation <a id="gae"></a>
- Use **λ return**
- n-step bootstrapping with n>1 often performs better
- However: It is still hard to tell what the number should be
- Idea of **λ return**: Create a mixture of all n-step bootstrapping estimates at once
- **λ** is a hyperparameter used for weighting the combination of each n-step estimate to the return
- Exponential decay of **λ return** with increasing n
- Example: **λ** = 0.5
- Calculating the **λ return** for state **S<sub>t</sub>**: Multiply all n-step returns with the corresponding weight and add them all up.
- For **λ** = 0: The lambda return is equivalent to the One-step TD estimate.
- For **λ** = 1: The lambda return is equivalent to the Infinte-step MC estimate.
- For 0 < **λ** <1: mixture of all n-step bootstrapping estimates

Keep in mind:
- This type of return can be combined with any policy-based method.
- Training is very quick because multiple value functions spread around on every time step.

- Link to the GAE paper: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

    ![image14]

## DDPG: Deep Deterministic Policy Gradient, Continuous Action-space <a id="ddpg_1"></a>
- Read the [DDPG Paper](https://arxiv.org/abs/1509.02971)
- DDPG is a different kind of actor-critic method.

However: It could be even **regarded as a DQN instead of an Actor Crtitc**.
- Reason: The critic in DDPG is used to approximate the maximizer over the Q values of the next state and not as a learned baseline.
- One limitation of DQN agent is that it is not straightforward to use in continuous action spaces.

Example (discrete action space):
- Imagine a DQN network that takes in a state and outputs the action value function.
- Imagine that there are five possible actions: Up and Down, left, right, jump
- **Q(s, "up")** gives you the estimated expected value for selecting the up action in state s (-2.18)
- **Q(s, "down")** gives you the estimated expected value for selecting the down action in state s (8.45)
- To find the max action value function for this state, you take the max of these values (9.12)

What if you have an action with continuous range?
- Imagine that the jump action is continuous (1...100cm)
- A DDPG can solve those tasks with continuous action spaces

    ![image15]

What is the DDPG architecture?
- Two neural networks
    - Actor network: used to approximate the **optimal policy deterministically**. We want to output the **best believed action** for any given state. It learns the **argmax<sub>a</sub>Q(s,a)**, which is the best action.
    - Critic network: learns to evaluate the **optimal action value function** by using the actors **best believed action**.
- The architecture is similar to DQN.

    ![image16]

## DDPG: Deep Deterministic Policy Gradient, Soft Updates <a id="ddpg_2"></a>
Two aspects of DDPG are interesting:
- The use of a replay buffer
- Soft updates to the target network

Update strategy in DDPG:
- In DDPG there are **two copies of the network weights** for each network:
    - A regular for the actor and a regular for the critic
    - A target for the actor and a target for the critic
- Soft update strategy consists of slowly blending regular network weights with your target network weights.
- You make target network be 99.99% of your target network weights and only 0.01% of your regular network weights.
- You slowly mix in your regular network weights into your target network weights.
- The regular network is the most up to date network because it is the one we are training. The target network is the one we use for prediction to stabilize strain.
- In practice, you'll get a faster convergence by using this update strategy.
- This target network update strategy can be used  with other algorithms that use target networks including DQN.

    ![image17]


## DDPG: Code Walk-through <a id="ddpg_code"></a>
- [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
- GitHub repo of [ShangtongZhang - Deep RL](https://github.com/ShangtongZhang/DeepRL)
- Video tutorial on YouTube [Video](https://www.youtube.com/watch?time_continue=7&v=LiUBJje2N0c&feature=emb_logo)

## DDPG: Code Implementation <a id="ddpg_impl"></a> 
- [Udacity DRL Github Repository: DDPG-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)

- Use setup instructions shown in repo [Deep Reinforcement Learning Project - Continuous Control](https://github.com/ddhartma/Deep-Reinforcement-Learning-Project-Continuous-Control#deep-reinforcement-learning-project---continuous-control)

### DDPG Notebook (Main File)
- Open Jupyter Notebook ```notebooks_python/DDPG.ipynb```

### DDPG Agent
- Open Jupyter Notebook ```notebooks_python/ddpg_agent.py```

### DDPG Model
- Open Jupyter Notebook ```notebooks_python/model.py```

## DDPG: Paper Walk-through <a id="ddpg_paper"></a> 

## Setup Instructions <a id="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a id="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a id="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Deep-Reinforcement-Learning-Theory-Actor-Critic-Methods.git
```

- Change Directory
```
$ cd Deep-Reinforcement-Learning-Theory-Actor-Critic-Methods
```

- Create a new Python environment, e.g. spark_env. Inside Git Bash (Terminal) write:
```
$ conda create --name drl_env
```

- Activate the installed environment via
```
$ conda activate drl_env
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a id="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Deep Reinforcement Learning'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a id="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Very good summary of DQN](https://medium.com/@nisheed/udacity-deep-reinforcement-learning-project-1-navigation-d16b43793af5)
* [An Introduction to Deep Reinforcement Learning](https://thomassimonini.medium.com/an-introduction-to-deep-reinforcement-learning-17a565999c0c)
* Helpful medium blog post on policies [Off-policy vs On-Policy vs Offline Reinforcement Learning Demystified!](https://kowshikchilamkurthy.medium.com/off-policy-vs-on-policy-vs-offline-reinforcement-learning-demystified-f7f87e275b48)
* [Understanding Baseline Techniques for REINFORCE](https://medium.com/@fork.tree.ai/understanding-baseline-techniques-for-reinforce-53a1e2279b57)
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Cheat Sheet](https://towardsdatascience.com/reinforcement-learning-cheat-sheet-2f9453df7651)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)

Important publications
* [2004 Y. Ng et al., Autonomoushelicopterflightviareinforcementlearning --> Inverse Reinforcement Learning](https://people.eecs.berkeley.edu/~jordan/papers/ng-etal03.pdf)
* [2004 Kohl et al., Policy Gradient Reinforcement Learning for FastQuadrupedal Locomotion --> Policy Gradient Methods](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/icra04.pdf)
* [2013-2015, Mnih et al. Human-level control through deep reinforcementlearning --> DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [2014, Silver et al., Deterministic Policy Gradient Algorithms --> DPG](http://proceedings.mlr.press/v32/silver14.html)
* [2015, Lillicrap et al., Continuous control with deep reinforcement learning --> DDPG](https://arxiv.org/abs/1509.02971)
* [2015, Schulman et al, High-Dimensional Continuous Control Using Generalized Advantage Estimation --> GAE](https://arxiv.org/abs/1506.02438)
* [2016, Schulman et al., Benchmarking Deep Reinforcement Learning for Continuous Control --> TRPO and GAE](https://arxiv.org/abs/1604.06778)
* [2017, PPO](https://openai.com/blog/openai-baselines-ppo/)
* [2018, Bart-Maron et al., Distributed Distributional Deterministic Policy Gradients](https://openreview.net/forum?id=SyZipzbCb)
* [2013, Sergey et al., Guided Policy Search --> GPS](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf)
* [2015, van Hasselt et al., Deep Reinforcement Learning with Double Q-learning --> DDQN](https://arxiv.org/abs/1509.06461)
* [1993, Truhn et al., Issues in Using Function Approximation for Reinforcement Learning](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf)
* [2015, Schaul et al., Prioritized Experience Replay --> PER](https://arxiv.org/abs/1511.05952)
* [2015, Wang et al., Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
* [2016, Silver et al., Mastering the game of Go with deep neural networks and tree search](https://www.researchgate.net/publication/292074166_Mastering_the_game_of_Go_with_deep_neural_networks_and_tree_search)
* [2017, Hessel et al. Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
* [2016, Mnih et al., Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
* [2017, Bellemare et al., A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
* [2017, Fortunato et al., Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
* [2016, Wang et al., Sample Efficient Actor-Critic with Experience Replay --> ACER](https://arxiv.org/abs/1611.01224)
* [2017, Lowe et al. Multi-Agent Actor-Critic for MixedCooperative-Competitive Environments](https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)
* [2017, Silver et al. Mastering the Game of Go without Human Knowledge --> AlphaGo Zero](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)
* [2017, Silver et al., Mastering Chess and Shogi by Self-Play with aGeneral Reinforcement Learning Algorithm --> AlphaZero](https://arxiv.org/pdf/1712.01815.pdf)

