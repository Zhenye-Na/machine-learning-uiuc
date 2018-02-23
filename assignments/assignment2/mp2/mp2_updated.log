test_forward_shape (test_model.ModelTests) ... ok
test_forward_zero (test_model.ModelTests) ... ok
test_loss_backward_no_loss (test_model.ModelTests) ... ok
test_loss_backward_no_reg (test_model.ModelTests) ... ok
test_loss_no_loss (test_model.ModelTests) ... ERROR
test_loss_no_reg (test_model.ModelTests) ... ERROR
test_no_loss_with_reg (test_model.ModelTests) ... ERROR
test_analytic (test_analytic.ModelTests) ... ERROR
test_first_row (test_io.IoToolsTests) ... ok
test_one_hot (test_io.IoToolsTests) ... ok
test_read_dataset_not_none (test_io.IoToolsTests) ... ok

======================================================================
ERROR: test_analytic (test_analytic.ModelTests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/cs446grader/autograder/mp2_programming/submissions/zna2/tests_autograder/test_analytic.py", line 14, in test_analytic
    train_eval_model.train_model_analytic(processed_dataset, model)
  File "/home/cs446grader/autograder/mp2_programming/submissions/zna2/train_eval_model.py", line 97, in train_model_analytic
    model.w = np.linalg.inv(x.T.dot(x) + model.w_decay_factor * np.ones((x.T.dot(x).shape))).dot(x.T).dot(y)
  File "/home/cs446grader/grading_env/lib/python3.4/site-packages/numpy/linalg/linalg.py", line 528, in inv
    ainv = _umath_linalg.inv(a, signature=signature, extobj=extobj)
  File "/home/cs446grader/grading_env/lib/python3.4/site-packages/numpy/linalg/linalg.py", line 89, in _raise_linalgerror_singular
    raise LinAlgError("Singular matrix")
numpy.linalg.linalg.LinAlgError: Singular matrix

----------------------------------------------------------------------
Ran 11 tests in 0.017s

FAILED (errors=1)
