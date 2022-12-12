import sys

import numpy as np

from main import load_json, train_classifier
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utils as ut

def main(train_file, model_path, setting, value):
    x_train, y_train, x_control_train = load_json(train_file)

    # X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
    x_train = ut.add_intercept(x_train)

    # x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, 0.7)

    # print >> sys.stderr, "First row:"
    # print >> sys.stderr, x_train[0,:], y_train[0], x_control_train

    if setting == 'gamma':
        mode = {"accuracy": 1, "gamma": float(value)}
    elif setting == 'c':
        mode = {"fairness": 1}
    elif setting == 'baseline':
        mode = {}
    else:
        raise Exception("Don't know how to handle setting %s" % setting)

    thresh = {}
    if setting == 'c':
        thresh = dict((k, float(value)) for (k, v) in list(x_control_train.items()))
        # print("Covariance threshold: %s" % thresh)

    # print("Will train classifier on %s %s-d points" % x_train.shape, file=sys.stderr)
    # print("Sensitive attribute: %s" % (x_control_train.keys(),), file=sys.stderr)
    sensitive_attrs = list(x_control_train.keys())
    w = train_classifier(x_train, y_train, x_control_train,
                         sensitive_attrs, mode,
                         thresh)

    # print("Model trained successfully.", file=sys.stderr)
    np.save(model_path, w)


if __name__ == '__main__':
    main(*sys.argv[1:])
    exit(0)
