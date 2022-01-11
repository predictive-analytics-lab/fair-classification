import sys
import json

import numpy as np

from main import load_json, predict
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utils as ut


def main(test_file, model_path, output_file):
    """

    Args:
        cons_type:
            0 for all misclassifications
            1 for FPR
            2 for FNR
            4 for both FPR and FNR
        tau: DCCP parameter, controls how much weight to put on the constraints,
             if the constraints are not satisfied, then increase tau -- default is DCCP val 0.005
        mu: DCCP parameter, controls the multiplicative factor by which the tau increases in each
            DCCP iteration -- default is the DCCP val 1.2
        eps: stopping criteria for the convex solver. check the CVXPY documentation for details.
             default for CVXPY is 1e-6
    """
    x_test, y_test, x_control_test = load_json(test_file)

    # X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
    x_test = ut.add_intercept(x_test)

    w = np.load(model_path)

    # print("Model trained successfully.", file=sys.stderr)

    predictions = predict(w, x_test).tolist()
    output_file = open(output_file, "w")
    json.dump(predictions, output_file)
    output_file.close()
