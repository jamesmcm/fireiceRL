## Conquering Fire 'n Ice: A Step-by-Step PPO Reinforcement Learning Guide

This comprehensive guide provides a detailed walkthrough for implementing a Proximal Policy Optimization (PPO) reinforcement learning agent to master the NES classic, *Fire 'n Ice* (also known as *Solomon's Key 2*). The agent will learn to play directly from pixel data obtained from the fceux emulator's VRAM, utilizing a Convolutional Neural Network (CNN) for visual processing. This guide will cover everything from setting up the environment to designing a sophisticated reward function that encourages efficient and strategic gameplay.

### Step 1: Setting the Stage - Environment and Communication

The foundation of this project lies in establishing a robust communication channel between the Python-based PPO agent and the fceux emulator. This will be achieved using a combination of Python libraries and a Lua script running within fceux.

**1.1. Software and Library Installation:**

*   **fceux Emulator:** Download and install the fceux NES emulator.
*   **Python:** Ensure you have a recent version of Python installed.
*   **Core Libraries:** Install the necessary Python libraries:
    *   `numpy` for numerical operations.
    *   `torch` or `tensorflow` for building and training the PPO agent's neural networks.
    *   `opencv-python` for image processing tasks like resizing and converting the game's VRAM data.
    *   `pyzmq` for inter-process communication between Python and Lua.

**1.2. Python-fceux Bridge:**

Leveraging existing projects that have successfully bridged Python and fceux is highly recommended. Frameworks like `fceux2openai` and `gym-nes-mario-bros` provide excellent starting points. These typically involve a Lua script that runs within fceux to read memory addresses, VRAM, and send controller inputs via sockets, and a Python wrapper to interact with this script.

**1.3. The Lua Connection:**

A custom Lua script will be the heart of the emulator-side interaction. This script will perform the following critical functions:

*   **Memory Monitoring:** Continuously read the crucial RAM addresses you've identified to track game state.
*   **VRAM Extraction:** Capture the raw pixel data from the VRAM.
*   **Controller Input:** Receive commands from the Python agent and translate them into controller inputs (left, right, A, B, start).
*   **Communication:** Send the collected game state information (VRAM and RAM values) to the Python script and receive actions in return, likely using ZeroMQ sockets.

### Step 2: Crafting the Brain - The PPO Agent

The PPO agent will consist of two main neural network components: a CNN-based feature extractor to understand the game's visuals and a policy/value network to make decisions.

**2.1. Convolutional Neural Network (CNN) for Vision:**

The agent will not have direct access to object positions or game logic. Instead, it will "see" the game through the fceux VRAM. A CNN is the ideal tool for this, as it can learn to identify key features like the player character, fires, blocks, and enemies directly from the pixel data.

A typical CNN architecture for this purpose might consist of:

*   Several convolutional layers with ReLU activation functions to extract spatial hierarchies of features.
*   Max-pooling layers to reduce the spatial dimensions of the feature maps.
*   A flattening layer to convert the 2D feature maps into a 1D vector.

**2.2. Policy and Value Networks:**

The output of the CNN will be fed into two separate fully connected networks:

*   **Policy Network (Actor):** This network will output the probabilities of taking each possible action (left, right, A, and potentially B and start for menu navigation).
*   **Value Network (Critic):** This network will estimate the expected cumulative reward from the current state, helping to guide the policy network's learning.

Proximal Policy Optimization is a policy gradient method that is known for its stability and sample efficiency, making it a strong choice for this complex task.

### Step 3: The Reward System - Motivating the Agent

A well-designed reward function is paramount to the success of any reinforcement learning agent. The goal is to provide clear and consistent feedback that guides the agent towards the desired behaviors. Based on the provided memory addresses, here is a detailed reward structure:

**3.1. Core Gameplay Rewards:**

*   **Extinguishing Fires:** A positive reward should be given each time the number of fires at memory address `$00AB` decreases. This is the primary objective of each level.
*   **Level Completion:** A significant positive reward should be given when a level is completed. This can be detected by checking if the value at address `$06A9` becomes zero.
*   **World Progression:** A substantial reward should be tied to changes in the world completion bitmasks in the `$0400` memory range. Specifically, reward the agent for increasing the values at these addresses, as this indicates the completion of multiple levels and worlds.

**3.2. Penalties and Incentives for Efficient Play:**

*   **Penalizing In-Menu Time:** To discourage the agent from lingering in menus, a small negative reward should be applied for every frame the agent is not in a level. This can be determined by checking the values at addresses `$0018`, `$001C`, and `$00D0`.
*   **Pause Awareness:** Addresses `$031D` and `$0321` are set to `1` while a level is paused. A mild penalty per paused frame can deter unnecessary pausing without preventing deliberate restarts.
*   **Death Detection:** Address `$0003` equals `8` during the death animation. Treating the rising edge of this flag as a terminal signal allows penalties (and optional resets) for failed attempts without interfering with normal level completion.
*   **Menu Entry Reward:** Addresses `$0324`/`$0328` light up on the level-select menu. A one-off reward after a reset encourages the policy to progress from the title/menu sequence into actual levels.
*   **Discouraging Re-entry of Completed Levels:** The agent needs to be penalized for entering a level that has already been completed. This can be achieved by checking the corresponding bit in the world completion bitmask (`$0400` range) before entering a level. If the bit is already set, a negative reward should be applied.
*   **Encouraging Exploration of Uncompleted Levels:** To nudge the agent towards new challenges, a small initial positive reward can be given upon entering a level that has not yet been completed.
*   **Level Select Detection:** Memory addresses `$0324` and `$0328` are set to `1` while the level-select menu is shown. These flags can be used to avoid accidental episode termination or to modulate penalties while navigating menus.

**3.3. Reward Function Implementation:**

The reward function will be implemented in the Python script. After each action, the script will receive the updated memory values from the Lua script and calculate the reward based on the logic outlined above.

### Step 4: The Training Loop - Learning to Play

The main training loop will orchestrate the interaction between the agent and the game environment. Here's a high-level overview of the process:

1.  **Initialization:** Launch the fceux emulator with the *Fire 'n Ice* ROM and the Lua script. The Python script will connect to the emulator.
2.  **Observation:** The Python script will receive the initial game state, including the VRAM pixel data and the relevant RAM values.
3.  **Preprocessing:** The VRAM data will be preprocessed (e.g., resized, converted to grayscale) before being fed into the agent's CNN.
4.  **Action Selection:** The PPO agent will process the preprocessed VRAM and other state information to select an action based on its current policy.
5.  **Execution:** The selected action will be sent to the Lua script, which will execute the corresponding controller input in the emulator.
6.  **Environment Step:** The emulator will advance by one frame.
7.  **New Observation and Reward:** The Python script will receive the new VRAM and RAM data. The reward function will be used to calculate the reward for the previous action.
8.  **Data Storage:** The agent will store the state, action, reward, and other relevant information for training.
9.  **PPO Update:** Periodically, the agent will use the collected data to update its policy and value networks using the PPO algorithm.
10. **Repeat:** This process will repeat for thousands or even millions of frames, allowing the agent to gradually learn and improve its gameplay.

### Step 5: Action Space and Menu Navigation

**5.1. In-Game Actions:**

For the core gameplay, the action space will be discrete and consist of:

*   `left`
*   `right`
*   `A` (for creating and pushing ice blocks)

**5.2. Menu Navigation:**

To handle menu screens, the action space will need to be expanded to include:

*   `B`
*   `start`

A state-based approach can be used to determine when to use the menu navigation actions. For example, by monitoring the `in-level` memory addresses, the agent can switch to a "menu mode" with the expanded action space when not in a level.

By following these steps and leveraging the provided memory addresses, you can successfully implement a PPO reinforcement learning agent capable of learning to play and master the challenging puzzles of *Fire 'n Ice*. This project combines the power of deep reinforcement learning with the intricacies of classic game emulation, offering a rewarding and educational experience.

---

### Implementation Notes (current status)

* FCEUX has been rebuilt against the system Lua 5.1 shared library so externally compiled rocks can load without ABI mismatches.
* Lua-side communication relies on the actively maintained `lua-zmq` bindings (falling back to `lzmq` when present). The bridge now polls the socket without blocking the emulator.
* The bridge exposes a `set_speed` command so training runs can switch FCEUX into `normal`, `turbo`, or `nothrottle` speed modes directly from the Python shell.
* Python uses Gymnasium-style wrappers, stacked grayscale observations, and a PPO agent with regular logging (`metrics.csv`, `reward_components.csv`, `events.jsonl`) and periodic checkpoints.
* Episode termination in the bridge is configurable: the agent can continue play after level clears/deaths or trigger resets via CLI flags. Detection relies on the in-level flags (`$0018/$001C/$00D0`), level-select flags (`$0324/$0328`), pause flags (`$031D/$0321`), and the death animation flag (`$0003 == 8`).
* The PPO trainer can optionally trigger full environment resets after a configurable number of updates without positive reward, preventing the agent from stalling on unsolved levels.
