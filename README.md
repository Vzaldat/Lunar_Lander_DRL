### Environment - Lundar Lander v3 OpenAI Gym

The Lunar Lander environment can be found here: -

The cell below gives random action endpoints in the Lunar Lander environment and their visualisation.

The environment a classic rocket trajectory optimization problem. According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off.
The following constitutes action space of the agent:

*   0: do nothing
*   1: fire left orientation engine
*   2: fire main orientation engine
*   3: fire right orientation engine

and the following constitutes of state space of the agent:

*   Coordinates of lander in x & y
*   linear velocities in x & y
*   angle
*   Angular Velocity
*   Two booleans that specify whether each leg has landed or not

Rewards For agent:
After every step a reward is granted. The total reward of an episode is the sum of the rewards for all the steps within that episode.

For each step, the reward:

* is increased/decreased the closer/further the lander is to the landing pad.

* is increased/decreased the slower/faster the lander is moving.

* is decreased the more the lander is tilted (angle not horizontal).

* is increased by 10 points for each leg that is in contact with the ground.

* is decreased by 0.03 points each frame a side engine is firing.

* is decreased by 0.3 points each frame the main engine is firing.

The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

An episode is considered solution when it crosses above 200 points.


An Episode is finished if:
* Lander crashes
* Lander gets outside of the viewport
* Lander is still

# Class Agent
### Hyperparameters
* `BUFFER_SIZE`:
* `BATCH_SIZE`: number of frames taken into observation
* `GAMMA`:
* `EPS_START`: Epsilon value at the start of training
* `EPS_END`: Epsilon value at the end of training
* `EPS_DECAY`: value of decay in value of epsilon during the training
* `TAU`: Soft update parameter
* `LR`: Learning rate of optimizer

### Parameters
* `state_size` and `action_size`: `state_size` and `action_size` of the environment are defined.
* `optimizer`: Adam optimizer for backward pass for policyNet
* `tau`: soft update parameter for stable updates of targetNet
* `memory`: Replay Memory class
* `seed`: random seed for sampling experiencces from replay memory
* `policyNet`: Neural network of class `DQN`, responsible for action selection and approximating Q-values,  
* `targetNet`: Neural network of class `DQN`, calculates the Q-values with help of `rewards` and `next_states`

### `act(state, eps=0)`
Function takes in `states` and model evaluation of `policyNet` for the particular state tensor. After the forward pass of `policyNet` DQN, we have action values from policyNet. Here, we have epsilon greedy method to pick action depending on chance.

### `learn(exp, gamma)`:
From the experience tuple, we have `states`, `actions`, `rewards`, `next_state` and `dones`.
We use `next_state` to get the action values for next possible states from `targetNet`. The target action values `q_target` is given by the equation as follows:
<center>$ Q(s, a) ← r + γ×Q(s`, a`)×(1 - dones) $</center>
where,

* $ Q(s, a) $ is expected action values `q_expected`
* $ \gamma $ is discount rate
* $ Q(s`, a`) $ is the action values from `targetNet` `DQN` for `next_states` input. `q_target`

loss calculation: The mean squared error loss between `q_expected` and `q_target`. After initializing gradients of the tensor, backpropogation of loss to get new gradients and it updates the network weights using optimzier. Finally, the `targetNet` is softly updated towards `policyNet` to track its weights, improving stability during training.

### `soft_update(policyNet, targetNet)`:
the `targetNet` parameters are changed by the following equation:
<center>$Θ_{targetNet} ← Θ_{policyNet}×τ+Θ_{targetNet}×(1-τ)$</center>
where,

* $Θ_{targetNet}$: parameters of `targetNet`
* $Θ_{policyNet}$: parameters of `policyNet`
* τ: soft update parameter

## class ReplayMem

### Parameters

* `action_size`: action size of the environment
* `batch_size`: used for sampling
* `seed`: random seeding for sampling experiences
* `buffer_size`: maximum length of the deque
* `memory`: deque of `experience` tuples
* `experience`: named tuple with field names as `state`, `action`, `reward`, `next_state`, `done`

### `push(state, action, reward, next_state, done)`
Add `experience` tuple is added to the memory deque after extracting the arguments.

### `sample()`
Taking a random sample from the memory deque of size `batch_size` and extracting `states`, `actions`, `rewards`, `next_states`, `dones` as a tuple.

### `__len__()`:
Length of memory deque.

## Training dqn
`dqn(n_eps=1000, max_t=500, agent=Agent(state_size=8, action_size=4, seed=0))`:
Tracking `rewards`, `training_loss` and `success_eps` from episode 1 to 1000 and maximum steps per episode is 500.

At the start of each episode, reset the environment and at each state, let the agent perform an action by `agent.select_act(state, eps)`, which is defined as an epsilon greedy method.
For the particular `action`, extract `next_state`, `reward`, `terminated`, `truncated` and define `done`=`terminated` or `truncated`. Use `agent.memory` to `push` the `state`, `action`, `reward`, `next_state`, `done` as arguments.

if the `agent.memory` stack length exceeds the `BATCH_SIZE`, Sample experiences `exp` from `agent.memory.sample()` and track loss by calling `agent.learn`.

if in the turn, parameter `done` is `True`, you can exit the loop and end the episode.

Save model weights for `policyNet` of `agent`.

![image](https://github.com/user-attachments/assets/d4f33692-a4d2-43aa-9d44-73b2e82d8d0c)
