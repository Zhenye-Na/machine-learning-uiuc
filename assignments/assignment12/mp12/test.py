"""Unit Tests examples for mp12."""

from q_learning import get_action_index, scale_down_epsilon, run_selected_action, compute_target_q
import unittest
import pong_game as game
import numpy as np
import cv2

# Frames over which to anneal epsilon.
EXPLORE = 5000.

# Final value of epsilon.
FINAL_EPSILON = 0.05

# Starting value of epsilon.
INITIAL_EPSILON = 1.0

# Number of valid actions.
ACTIONS = 3

# Decay rate of past observations.
GAMMA = 0.99


class QlearningTest(unittest.TestCase):

    def test_1(self):
        epsilon = 0.01
        t = 2000
        out = scale_down_epsilon(epsilon, t)
        self.assertEqual(out, epsilon)

    def test_2(self):
        action_index_list = []

        for i in range(10):
            readout_t = [5, 2, 7, 18, 3]
            epsilon = 0.1
            t = 7000
            action_index = get_action_index(readout_t, epsilon, t)
            action_index_list.append(action_index)

        self.assertGreaterEqual(action_index_list.count(3), 6)

    def test_3(self):
        r_batch = [5]
        readout_j1_batch = [[1, 2, 3, 4]]
        terminal_batch = [True]
        target_q_batch = compute_target_q(
            r_batch, readout_j1_batch, terminal_batch)
        self.assertEqual(target_q_batch[0], r_batch[0])
