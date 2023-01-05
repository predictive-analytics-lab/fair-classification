import json
import sys
from typing import Optional

import numpy as np

import src.fair_classification.funcs_disp_mist as fdm
import src.fair_classification.utils as ut
from src.types import ConsParams, Mode, SensitiveAttrsToCovThresh


def train_classifier(x, y, control, cons_params: ConsParams, eps: float):
    loss_function = "logreg"  # only logistic regression is implemented
    w = fdm.train_model_disp_mist(x, y, control, loss_function, eps, cons_params)
    return w


def predict(model, x):
    return np.sign(np.dot(x, model))


def load_json(filename: str):
    f = json.load(open(filename))
    x = np.array(f["x"])
    y = np.array(f["class"])
    sensitive = {k: np.array(v) for (k, v) in f["sensitive"].items()}
    return x, y, sensitive


def main(
    train_file: str,
    test_file: str,
    output_file: str,
    mode: Mode,
    tau="5.0",
    mu="1.2",
    eps="0.0001",
):
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
    x_test, _, _ = load_json(test_file)

    # X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
    x_train = ut.add_intercept(x_train)
    x_test = ut.add_intercept(x_test)

    # x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, 0.7)

    # print >> sys.stderr, "First row:"
    # print >> sys.stderr, x_train[0,:], y_train[0], x_control_train

    sensitive_attrs = list(x_control_train.keys())
    sensitive_attr = str(sensitive_attrs[0])

    sensitive_attrs_to_cov_thresh: SensitiveAttrsToCovThresh = {
        sensitive_attr: {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}, 2: {0: 0, 1: 0}}
    }

    cons_params: Optional[ConsParams]
    if mode == "baseline":
        cons_params = None
    else:
        if mode == "fpr":
            cons_type = 1
        elif mode == "fnr":
            cons_type = 2
        elif mode == "fprfnr":
            cons_type = 4
        else:
            raise Exception("Don't know how to handle setting %s" % mode)

        cons_params = {
            "tau": float(tau),
            "mu": float(mu),
            "sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh,
            "cons_type": cons_type,
        }

    # print("Will train classifier on %s %s-d points" % x_train.shape, file=sys.stderr)
    # print("Sensitive attribute: %s" % (x_control_train.keys(),), file=sys.stderr)
    sensitive_attrs = list(x_control_train.keys())
    w = train_classifier(x_train, y_train, x_control_train, cons_params, float(eps))

    # print("Model trained successfully.", file=sys.stderr)

    predictions = predict(w, x_test).tolist()
    output_fp = open(output_file, "w")
    json.dump(predictions, output_fp)
    output_fp.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
    exit(0)
