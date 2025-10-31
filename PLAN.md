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

Leveraging existing projects that have successfully bridged Python and fceux is highly recommended. Frameworks like `fceux2openai` and `gym-nes-mario-bros` provide excellent starting points. Our bridge extends these ideas with a dedicated level-loader command: the Lua script listens for reset requests, applies a savestate, writes the target world/level directly into RAM, and resumes play without ever showing menus to the agent.

**1.3. The Lua Connection:**

A custom Lua script will be the heart of the emulator-side interaction. This script will perform the following critical functions:

*   **Memory Monitoring:** Continuously read the crucial RAM addresses you've identified to track game state.
*   **VRAM Extraction:** Capture the raw pixel data from the VRAM.
*   **Controller Input:** Receive commands from the Python agent and translate them into on-level controller inputs (left, right, A); menu buttons are no longer required because resets skip straight into gameplay.
*   **Level Injection:** Handle `reset`/`restart_level` commands from Python by restoring a clean savestate and force-writing the requested world/level so the agent spawns directly in the stage.
*   **Communication:** Send the collected game state information (VRAM and RAM values) to the Python script and receive actions in return, likely using ZeroMQ sockets.
*   **Port Coordination:** Read `FIREICE_PORT` (and optional fallback settings) from the environment so multiple Lua bridges can bind to distinct TCP ports without clashing.

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

*   **Policy Network (Actor):** This network will output the probabilities of taking each possible in-level action (left, right, A).
*   **Value Network (Critic):** This network will estimate the expected cumulative reward from the current state, helping to guide the policy network's learning.

Proximal Policy Optimization is a policy gradient method that is known for its stability and sample efficiency, making it a strong choice for this complex task.

### Step 3: The Reward System - Motivating the Agent

A well-designed reward function is paramount to the success of any reinforcement learning agent. The goal is to provide clear and consistent feedback that guides the agent towards the desired behaviors. Based on the provided memory addresses, here is a detailed reward structure:

**3.1. Core Gameplay Rewards:**

*   **Extinguishing Fires:** A positive reward should be given each time the number of fires at memory address `$00AB` decreases. This is the primary objective of each level.
*   **Level Completion:** A significant positive reward should be given when a level is completed. This can be detected by checking if the value at address `$06A9` becomes zero.
*   **World Progression:** A substantial reward should be tied to changes in the world completion bitmasks in the `$0400` memory range. Specifically, reward the agent for increasing the values at these addresses, as this indicates the completion of multiple levels and worlds.

**3.2. Penalties and Incentives for Efficient Play:**

*   **Level Entry Guardrails:** Because resets inject the agent directly into a stage, any time spent out of level is unexpected. Track `$0018`, `$001C`, and `$00D0` as assertions rather than as a reward component so you can raise alerts if the loader ever drops the agent back to a menu.
*   **Pause Awareness:** Addresses `$031D` and `$0321` are set to `1` while a level is paused. A mild penalty per paused frame can deter unnecessary pausing without preventing deliberate restarts.
*   **Death Detection:** Address `$0003` equals `8` during the death animation. Treating the rising edge of this flag as a terminal signal allows penalties (and optional resets) for failed attempts without interfering with normal level completion.
*   **Completed Level Penalty:** Penalize attempts to re-enter a cleared level by checking the matching bit in the world completion bitmask (`$0400` range) before allowing a jump.
*   **Exploration Incentive:** To nudge the agent towards new challenges, provide a small initial positive reward when dropping into a level that has not yet been completed.
*   **Menu State Monitoring:** Memory addresses `$0324` and `$0328` still toggle on the level-select menu; log them for diagnostics but avoid rewarding or penalizing them since the loader should keep the agent away from menus entirely.

**3.3. Reward Function Implementation:**

The reward function will be implemented in the Python script. After each action, the script will receive the updated memory values from the Lua script and calculate the reward based on the logic outlined above.

### Step 4: The Training Loop - Learning to Play

The main training loop will orchestrate the interaction between the agent and the game environment. Here's a high-level overview of the process:

1.  **Initialization:** Launch the fceux emulator with the *Fire 'n Ice* ROM and the Lua bridge script. When Python sends its first `reset`, the Lua side restores the configured savestate, overwrites the target world/level bytes, and drops the agent straight into the requested stage.
2.  **Observation:** The Python script will receive the initial game state, including the VRAM pixel data and the relevant RAM values.
3.  **Preprocessing:** The VRAM data will be preprocessed (e.g., resized, converted to grayscale) before being fed into the agent's CNN.
4.  **Action Selection:** The PPO agent will process the preprocessed VRAM and other state information to select an action based on its current policy.
5.  **Execution:** The selected action will be sent to the Lua script, which will execute the corresponding controller input in the emulator.
6.  **Environment Step:** The emulator will advance by one frame.
7.  **New Observation and Reward:** The Python script will receive the new VRAM and RAM data. The reward function will be used to calculate the reward for the previous action.
8.  **Data Storage:** The agent will store the state, action, reward, and other relevant information for training.
9.  **PPO Update:** Periodically, the agent will use the collected data to update its policy and value networks using the PPO algorithm.
10. **Repeat:** This process will repeat for thousands or even millions of frames, allowing the agent to gradually learn and improve its gameplay.

#### Parallel Environment Batching

To accelerate data collection, the trainer can drive multiple FCEUX instances concurrently (`--num-workers`). Observations, actions, value targets, and advantages are batched across workers each rollout step. Episodes terminate independently: when a worker reports `done`, the trainer records its terminal statistics, immediately issues an environment reset on that worker, and continues populating the batch with the fresh observation. Reward accounting and stagnation logic aggregate totals across all workers, while CNN snapshots write into per-worker subdirectories to avoid filename clashes.

### Step 5: Action Space and Menu Handling

**5.1. Core Actions:**

For the core gameplay, the discrete action space covers what the agent needs on every stage:

*   `noop`
*   `left`
*   `right`
*   `A` (creating and pushing ice blocks)

**5.2. Script-Driven Resets:**

The Lua loader is responsible for every transition back into a level. Resets and stagnation restarts always restore the savestate, write the desired world/level, and resume the stage. Because of this automation the agent never presses `start` or `B`, and any appearance of menus should be treated as a bug in the loader rather than as behaviour to be learned.

By following these steps and leveraging the provided memory addresses, you can successfully implement a PPO reinforcement learning agent capable of learning to play and master the challenging puzzles of *Fire 'n Ice*. This project combines the power of deep reinforcement learning with the intricacies of classic game emulation, offering a rewarding and educational experience.

---

### Implementation Notes (current status)

* FCEUX has been rebuilt against the system Lua 5.1 shared library so externally compiled rocks can load without ABI mismatches.
* Lua-side communication relies on the actively maintained `lua-zmq` bindings (falling back to `lzmq` when present). The bridge now polls the socket without blocking the emulator.
* The bridge exposes a `set_speed` command so training runs can switch FCEUX into `normal`, `turbo`, or `nothrottle` speed modes directly from the Python shell.
* Python uses Gymnasium-style wrappers, stacked grayscale observations, and a PPO agent with regular logging (`metrics.csv`, `reward_components.csv`, `events.jsonl`) and periodic checkpoints.
* A lightweight launcher spawns N FCEUX processes with consecutive ports (`--base-port`), exports `FIREICE_PORT` for each worker, and cleans them up when training stops.
* Episode termination in the bridge is configurable: the agent can continue play after level clears/deaths or trigger resets via CLI flags. Detection relies on the in-level flags (`$0018/$001C/$00D0`), level-select flags (`$0324/$0328`), pause flags (`$031D/$0321`), and the death animation flag (`$0003 == 8`).
* The PPO trainer can optionally trigger full environment resets after a configurable number of updates without positive reward, preventing the agent from stalling on unsolved levels.
