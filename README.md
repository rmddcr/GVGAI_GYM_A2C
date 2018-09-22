# INSTRUCTIONS FOR UCSC 

## Requirements
* user python 3.6.6 or 3.5.*
* for mac users if matplotlib is giving trouble [Check this link](https://matplotlib.org/faq/osx_framework.html#osxframework-faq)
  - If there is no visible plot add `backend : macosx` to  **~./.matplotlib/matplotlibrc**

## Usage 
* Build project `pip install -e <package-location>`
* Install any missing libraries with pip
* Run agent `python RunMe.py`

## References
* [Simple Reinforcement Learning with Tensorflow Part 8: Asynchronous Actor-Critic Agents (A3C)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)

___

# GVGAI GYM

An [OpenAI Gym](gym.openai.com) environment for games written in the [Video Game Description Language](http://www.gvgai.net/vgdl.php), including the [Generic Video Game Competition](http://www.gvgai.net/) framework. The framework, along with some initial reinforcement learning results, is covered in the paper [Deep Reinforcement Learning for General Video Game AI](https://arxiv.org/abs/1806.02448).

## Installation

- Clone this repository to your local machine.
- To install the package, run `pip install -e <package-location>`
  (This should install OpenAI Gym automatically, otherwise it can be installed [here](https://github.com/openai/gym)
- Install a Java compiler `javac` (e.g. `sudo apt install openjdk-9-jdk-headless`)

## Usage

Demo video on [YouTube](https://youtu.be/O84KgRt6AJI)

Once installed, it can be used like any OpenAI Gym environment.

Run the following line to get a list of all GVGAI environments.
```Python
[env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai')]
```

## Resources

[GVGAI website](http://www.gvgai.net)

[GVGAI-Gym (master branch)](https://github.com/rubenrtorrado/GVGAI_GYM) 

[Demo video on YouTube](https://youtu.be/O84KgRt6AJI)
