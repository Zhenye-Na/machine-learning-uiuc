# Assignment 12 - Q Learning

![](https://github.com/Zhenye-Na/cs446/blob/master/assignments/assignment12/mp12/pong.gif?raw=true)

```
test_compute_target_q1 (test_autograder.QlearningTest) ... ok
test_compute_target_q2 (test_autograder.QlearningTest) ... ERROR
test_get_action_index1 (test_autograder.QlearningTest) ... ok
test_get_action_index2 (test_autograder.QlearningTest) ... ok
test_run_selected_action1 (test_autograder.QlearningTest) ... ok
test_run_selected_action2 (test_autograder.QlearningTest) ... ok
test_scale_down_epsilon1 (test_autograder.QlearningTest) ... ok
test_scale_down_epsilon2 (test_autograder.QlearningTest) ... ok

======================================================================
ERROR: test_compute_target_q2 (test_autograder.QlearningTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/cs446grader/autograder/mp12_programming/submissions/zna2/tests_autograder/tests_autograder/test_autograder.py", line 130, in test_compute_target_q2
    target_q_batch = compute_target_q(r_batch,readout_j1_batch,terminal_batch)
  File "/home/cs446grader/autograder/mp12_programming/submissions/zna2/q_learning.py", line 217, in compute_target_q
    r_batch[i] + GAMMA * readout_j1_batch[i].max())
AttributeError: 'list' object has no attribute 'max'

----------------------------------------------------------------------
Ran 8 tests in 0.046s

FAILED (errors=1)
```
