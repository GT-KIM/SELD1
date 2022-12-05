# Extracts the features, labels, and normalizes the development and evaluation split features.

import scripts.cls_feature_class as cls_feature_class
import config.parameters as parameters
import sys


def main():
    # Expects one input - task-id - corresponding to the configuration given in the parameter.py file.
    # Extracts features and labels relevant for the task-id
    # It is enough to compute the feature and labels once. 

    params = parameters.get_params()

    # -------------- Extract features and labels for development set -----------------------------
    dev_feat_cls = cls_feature_class.FeatureClass(params)

    # # Extract features and normalize them
    dev_feat_cls.extract_all_feature()
    dev_feat_cls.preprocess_features()

    # # Extract labels
    dev_feat_cls.extract_all_labels()

if __name__ == "__main__":
    main()

