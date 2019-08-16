import os, sys
import numpy as np
import json

sys.path.insert(
    0, "../../fair_classification/"
)  # the code for fair classification is in this directory
import utils as ut
import funcs_disp_mist as fdm


def predict(model, x):
    return np.sign(np.dot(x, model))


def load_json(filename):
    f = json.load(open(filename))
    x = np.array(f["x"])
    y = np.array(f["class"])
    sensitive = dict((k, np.array(v)) for (k, v) in f["sensitive"].items())
    return x, y, sensitive


def main(train_file, test_file, output_file, cons_type=1, tau=5.0, mu=1.2):
    """

    Args:
        cons_type: 1: FPR constraint
    """

    x_train, y_train, x_control_train = load_json(train_file)
    x_test, y_test, x_control_test = load_json(test_file)

    x_train = ut.add_intercept(x_train)
    x_test = ut.add_intercept(x_test)

    loss_function = "logreg"  # perform the experiments with logistic regression
    EPS = 1e-6

    unconstrained = False
    if unconstrained:
        cons_params = None  # constraint parameters, will use them later
        # == Unconstrained (original) classifier ==
        w = fdm.train_model_disp_mist(
            x_train, y_train, x_control_train, loss_function, EPS, cons_params
        )

    else:
        # Now classify such that we optimize for accuracy while achieving perfect fairness
        # zero covariance threshold, means try to get the fairest solution
        sensitive_attrs_to_cov_thresh = {
            "race": {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}, 2: {0: 0, 1: 0}}
        }
        cons_params = {
            "cons_type": cons_type,
            "tau": tau,
            "mu": mu,
            "sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh,
        }

        w = fdm.train_model_disp_mist(
            x_train, y_train, x_control_train, loss_function, EPS, cons_params
        )

    predictions = predict(w, x_test).tolist()
    output_file = open(output_file, "w")
    json.dump(predictions, output_file)
    output_file.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
    exit(0)
