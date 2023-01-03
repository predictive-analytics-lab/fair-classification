import sys

import numpy as np
from .main import load_json, train_classifier

import src.fair_classification.utils as ut


def main(train_file, model_path, mode, tau="5.0", mu="1.2", eps="0.0001"):
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
    x_train, y_train, x_control_train = load_json(train_file)

    # X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
    x_train = ut.add_intercept(x_train)

    # x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, 0.7)

    # print >> sys.stderr, "First row:"
    # print >> sys.stderr, x_train[0,:], y_train[0], x_control_train

    sensitive_attrs = list(x_control_train.keys())
    sensitive_attr = str(sensitive_attrs[0])

    sensitive_attrs_to_cov_thresh = {
        sensitive_attr: {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}, 2: {0: 0, 1: 0}}
    }
    cons_params = {
        "tau": float(tau),
        "mu": float(mu),
        "sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh,
    }

    if mode == "fpr":
        cons_type = 1
        cons_params["cons_type"] = cons_type
    elif mode == "fnr":
        cons_type = 2
        cons_params["cons_type"] = cons_type
    elif mode == "fprfnr":
        cons_type = 4
        cons_params["cons_type"] = cons_type
    elif mode == "baseline":
        cons_params = None
    else:
        raise Exception("Don't know how to handle setting %s" % mode)

    # print("Will train classifier on %s %s-d points" % x_train.shape, file=sys.stderr)
    # print("Sensitive attribute: %s" % (x_control_train.keys(),), file=sys.stderr)
    sensitive_attrs = list(x_control_train.keys())
    w = train_classifier(x_train, y_train, x_control_train, cons_params, float(eps))

    # print("Model trained successfully.", file=sys.stderr)
    np.save(model_path, w)


if __name__ == "__main__":
    main(*sys.argv[1:])
    exit(0)
