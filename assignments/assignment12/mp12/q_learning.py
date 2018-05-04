
import tensorflow as tf
import cv2
import pong_game as game
import random
import numpy as np
from collections import deque

# Game name.
GAME = 'Pong'

# Number of valid actions.
ACTIONS = 3

# Decay rate of past observations.
GAMMA = 0.99

# Timesteps to observe before training.
OBSERVE = 5000.

# Frames over which to anneal epsilon.
EXPLORE = 5000.

# Final value of epsilon.
FINAL_EPSILON = 0.05

# Starting value of epsilon.
INITIAL_EPSILON = 1.0

# Number of previous transitions to remember in the replay memory.
REPLAY_MEMORY = 590000

# Size of minibatch.
BATCH = 32

# Only select an action every Kth frame, repeat the same action
# for other frames.
K = 2

# Learning Rate.
Lr = 1e-6


def weight_variable(shape):
    """Initialize the weight variable."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    """Initializa the bias variable."""
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    """Define a convolutional layer."""
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    """Define a maxpooling layer."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def createNetwork():
    """Create a convolutional network for estimating the Q value.

    Args:
    Returns:
        s: Input layer
        readout: Output layer with the Q-values for every possible action
    """
    # Initialize the network weights and biases.
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # Input layer.
    s = tf.placeholder("float", [None, 80, 80, 4])

    # Hidden layers.
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # Output layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout


def get_action_index(readout_t, epsilon, t):
    """Choose an action epsilon-greedily.

    Details:
        choose an action randomly:
        (1) in the observation phase (t < OBSERVE).
        (2) beyond the observation phase with probability "epsilon".
        otherwise, choose the action with the highest Q-value.
    Args:
        readout_t: a vector with the Q-value associated with every action.
        epsilon: temperature variable for exploration-exploitation.
        t: current number of iterations.
    Returns:
        index: the index of the action to be taken next.
    """
    p = random.random()
    if t < OBSERVE:
        action_index = random.randint(0, len(readout_t) - 1)
    else:
        if p >= epsilon:
            action_index = np.argmax(readout_t)
        else:
            action_index = random.randint(0, len(readout_t) - 1)

    return action_index


def scale_down_epsilon(epsilon, t):
    """Epsilon decrease.

    Decrease epsilon after by ((INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE)
    in case epsilon is larger than the desired final epsilon or beyond
    the observation phase.
    Args:
        epsilon: the current value of epsilon.
        t: current number of iterations.
    Returns:
        the updated epsilon
    """
    if (t > OBSERVE) and (epsilon > FINAL_EPSILON):
        epsilon -= ((INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE)
    return epsilon


def run_selected_action(a_t, s_t, game_state):
    """Run the selected action and return the next state and reward.

    Do not forget that state is composed of the 4 previous frames.
    Hint: check the initialization for the interface to the game simulator.

    Args:
        a_t: current action.
        s_t: current state.
        game_state: game state to communicate with emulator.
    Returns:
        s_t1: next state.
        r_t: reward.
        terminal: indicating whether the episode terminated (output of the simulator).
    """
    x_t, r_t, terminal = game_state.frame_step(a_t)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t1 = np.stack((s_t[:, :, 1], s_t[:, :, 2], s_t[:, :, 3], x_t), axis=2)

    return s_t1, r_t, terminal


def compute_cost(target_q, a_t, q_value):
    """Compute the cost.

    Args:
        target_q: target Q-value.
        a_t: current action.
        q_value: current Q-value.
    Returns:
        cost
    """
    # Q-value for the action.
    readout_action = tf.reduce_sum(
        tf.multiply(q_value, a_t), reduction_indices=1)

    # Q-Learning Cost.
    cost = tf.reduce_mean(tf.square(target_q - readout_action))

    return cost


def compute_target_q(r_batch, readout_j1_batch, terminal_batch):
    """Compute the target Q-value for all samples in the batch.

    Distinguish two cases:
    1. The next state is a terminal state.
    2. The next state is not a terminal state.
    Args:
        r_batch: batch of rewards.
        readout_j1_batch: batch of Q-values associated with the next state.
        terminal_batch: batch of boolean variables indicating the game termination.
    Returns:
        target_q_batch: batch of target Q values.

    Hint: distinguish two cases: (1) terminal state and (2) non terminal states
    """
    target_q_batch = []

    for i in range(0, len(terminal_batch)):
        # If the terminal state is reached, the Q-value is only equal to the reward.
        if terminal_batch[i]:
            target_q_batch.append(r_batch[i])
        else:
            target_q_batch.append(
                r_batch[i] + GAMMA * readout_j1_batch[i].max())

    return target_q_batch


def trainNetwork(s, readout, sess):
    """Train the artificial agent using Q-learning to play the pong game.

    Args:
        s: the current state formed by 4 frames of the playground.
        readout: the Q value for each passible action in the current state.
        sess: session
    """
    # Placeholder for the action.
    a = tf.placeholder("float", [None, ACTIONS])

    # Placeholder for the target Q value.
    y = tf.placeholder("float", [None])

    # Compute the loss.
    cost = compute_cost(y, a, readout)

    # Training operation.
    train_step = tf.train.AdamOptimizer(Lr).minimize(cost)

    # Open up a game state to communicate with emulator.
    game_state = game.GameState()

    # Initialize the replay memory.
    D = deque()

    # Initialize the action vector.
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    # Initialize the state of the game.
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # Save and load model checkpoints.
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("./saved_networks_q_learning/")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # Initialize the epsilon value for the exploration phase.
    epsilon = INITIAL_EPSILON

    # Initialize the iteration counter.
    t = 0

    while True:
        # Choose an action epsilon-greedily.
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]

        action_index = get_action_index(readout_t, epsilon, t)

        a_t = np.zeros([ACTIONS])

        a_t[action_index] = 1

        # Scale down epsilon during the exploitation phase.
        epsilon = scale_down_epsilon(epsilon, t)

        # Run the selected action and update the replay memeory
        for i in range(0, K):
            # Run the selected action and observe next state and reward.
            s_t1, r_t, terminal = run_selected_action(a_t, s_t, game_state)

            # Store the transition in the replay memory D.
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

        # Start training once the observation phase is over.
        if (t > OBSERVE):

            # Sample a minibatch to train on.
            minibatch = random.sample(list(D), BATCH)

            # Get the batch variables.
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]
            terminal_batch = [d[4] for d in minibatch]

            # Compute the target Q-Value
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
            target_q_batch = compute_target_q(
                r_batch, readout_j1_batch, terminal_batch)

            # Perform gradient step.
            train_step.run(feed_dict={
                y: target_q_batch,
                a: a_batch,
                s: s_j_batch})

        # Update the state.
        s_t = s_t1

        # Update the number of iterations.
        t += 1

        # Save a checkpoint every 10000 iterations.
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks_q_learning/' +
                       GAME + '-dqn', global_step=t)

        # Print info.
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION",
              action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))


def playGame():
    """Paly the pong game."""
    # Start an active session.
    sess = tf.InteractiveSession()

    # Create the network.
    s, readout = createNetwork()

    # Q-Learning
    s, readout = trainNetwork(s, readout, sess)


def main():
    """Main function."""
    playGame()


if __name__ == "__main__":
    main()
