test_inf (test_inference.InferenceTests) ... ok
test_local_score (test_inference.InferenceTests) ... FAIL
test_pairwise_potentials (test_potentials.PotentialTests) ... 2018-03-12 09:42:02.138948: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
ok
test_unary_potentials (test_potentials.PotentialTests) ... ok
test_learning_obj (test_learning.LearningTests) ... ok
test_belief_convergence (test_beliefs.BeliefTests) ... ok
test_pairwise_beliefs (test_beliefs.BeliefTests) ... ok
test_pairwise_features (test_features.FeatureTests) ... ok
test_unary_features (test_features.FeatureTests) ... ok

======================================================================
FAIL: test_local_score (test_inference.InferenceTests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/cs446grader/autograder/mp7_programming/submissions/zna2/tests_autograder/test_inference.py", line 82, in test_local_score
    self.assertEqual(result, 0)
AssertionError: 4 != 0

----------------------------------------------------------------------
Ran 9 tests in 0.140s

FAILED (failures=1)
