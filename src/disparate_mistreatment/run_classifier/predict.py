import sys
import json

import numpy as np

from main import load_json, predict

import src.fair_classification.utils as ut


def main(test_file, model_path, output_file):
    x_test, y_test, x_control_test = load_json(test_file)

    # X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
    x_test = ut.add_intercept(x_test)

    w = np.load(model_path)

    predictions = predict(w, x_test).tolist()
    output_file = open(output_file, "w")
    json.dump(predictions, output_file)
    output_file.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
    exit(0)
