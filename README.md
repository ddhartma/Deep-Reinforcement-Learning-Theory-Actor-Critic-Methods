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
- [GAE: Generalized Advantage Estimation](#gae)
- [DDPG: Deep Deterministic Policy Gradient, Continuous Action-space](#ddpg_1)
- [DDPG: Deep Deterministic Policy Gradient, Soft Updates](#ddpg_2)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Introduction <a name="what_is_reinforcement"></a>
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


## Motivation <a name="mot"></a> 
- Methods that **learn both policies and value functions** are referred to as **actor-critic** because the **policy, which selects actions,** can be seen as an **actor**, and the **value function, which evaluates policies,** can be seen as a **critic**. 
- Actor-critic methods often **perform better than value-based or policy-gradient methods** alone on many of the deep reinforcement learning benchmarks.

    ![image1]

### Actor-critic methods: 
- Use a value function as a baseline
- Train a **neural network to approximate a value function** and then **use it as a baseline**
- All actor-critic methods use **value-based techniques** to further **reduce the variance of policy-based methods**.

## Bias and Variance <a name="bias"></a> 
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

## Two Ways For Estimating Expected Returns <a name="exp_ret"></a> 
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




## Baselines and Critics <a name="baselines"></a> 

## Policy-based, Value-based and Actor-Critic <a name="actor_critic"></a> 

## A Basis Actor-Critic Agent <a name="basis"></a> 

## A3C: Asynchronous Advantage Actor-Critic, N-step Bootstrapping <a name="a3c_1"></a> 

## A3C: Asynchronous Advantage Actor-Critic, Parallel Training <a name="a3c_2"></a> 

## A3C: Asynchronous Advantage Actor-Critic, Off-policy vs. On-policy <a name="a3c_3"></a> 

## A2C: Advantage Actor-Critic <a name="a2c"></a> 

## GAE: Generalized Advantage Estimation <a name="gae"></a> 

## DDPG: Deep Deterministic Policy Gradient, Continuous Action-space <a name="ddpg_1"></a> 

## DDPG: Deep Deterministic Policy Gradient, Soft Updates <a name="ddpg_2"></a> 


## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
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

### Clone the project <a name="Clone_the_project"></a>
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

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Deep Reinforcement Learning'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

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
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)
