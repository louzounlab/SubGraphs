import nni
import logging
from torch.optim import Adam, SGD
import os
import argparse
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../graph_calculations/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_algorithms/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_algorithms/accelerated_graph_features/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_algorithms/vertices/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_infra/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/graph_infra/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_processor/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_infra/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_meta/'))

from subgraphs_GCN import GCN_subgraphs

logger = logging.getLogger("NNI_logger")


# TODO: CHECK WHICH PARAMS ARE NEEDED FOR THE MODEL
def run_trial(params, v, p, cs, d):

    #features = params["input_vec"]

    # model
    hid_features = params["hid_features"]
    dropout = params["dropout"]
    reg_term = params["regularization"]
    lr = params["learning_rate"]
    optimizer = Adam if params["optimizer"] == "Adam" else SGD
    epochs = int(params["epochs"])


    input_params = {
        "hid_features": hid_features,
        "epochs": epochs,
        "dropout": dropout,
        "lr": lr,
        "regularization": reg_term,
        "optimizer": optimizer

    }
    #model = GCNTemporalCommunities(v, p, cs, d, features=features, norm_adj=True, nni=True)
    model = GCN_subgraphs(nni=True)
    model.train(input_params)


def main(v, p, cs, d):
    try:
        # get parameters form tuner
        params = nni.get_next_parameter()
        logger.debug(params)
        run_trial(params, v, p, cs, d)
    except Exception as exception:
        logger.error(exception)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int)
    parser.add_argument("-p", type=float, default=0.5)
    parser.add_argument("-cs", type=int)
    parser.add_argument("-d", type=bool, default=False)
    args = vars(parser.parse_args())
    main(args['n'], args['p'], args['cs'], args['d'])
