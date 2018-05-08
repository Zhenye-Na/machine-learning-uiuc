test_d_loss_o (tests_autograder.DeepLearningTest) ... ok
test_derivative_of_e (tests_autograder.DeepLearningTest) ... ok
test_derivative_of_f (tests_autograder.DeepLearningTest) ... FAIL
test_derivative_of_u (tests_autograder.DeepLearningTest) ... ok
test_derivative_of_w (tests_autograder.DeepLearningTest) ... ok
test_error_at_layer1 (tests_autograder.DeepLearningTest) ... ok
test_error_at_layer2 (tests_autograder.DeepLearningTest) ... ok
test_forward (tests_autograder.DeepLearningTest) ... ok
test_sigmoidPrime (tests_autograder.DeepLearningTest) ... ok

======================================================================
FAIL: test_derivative_of_f (tests_autograder.DeepLearningTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/cs446grader/autograder/mp6_programming/submissions/zna2/tests_autograder/tests_autograder.py", line 175, in test_derivative_of_f
    self.assertAlmostEqual(sol_student[0], sol[0])
AssertionError: 13 != 6.5 within 7 places

----------------------------------------------------------------------
Ran 9 tests in 0.007s

FAILED (failures=1)
