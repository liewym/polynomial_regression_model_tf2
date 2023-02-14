# polynomial_regression_model_tf2

This tensorflow model is used to segment the lumen in IVOCT images. The input of the model needed to be a polar IVOCT images. The outputs are 50 sets of polynomial coefficients of lumen line fragment. Note that the output from the model should be applied with polynomial_fit function (in training&testing.py) to convert them from polynomial coefficients to a continuous lumen line (radial distance points).
