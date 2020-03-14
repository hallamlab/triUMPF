__author__ = "Abdurrahman M. A. Basher"
__date__ = '25/10/2019'
__copyright__ = "Copyright 2019, The Hallam Lab"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Abdurrahman M. A. Basher"
__email__ = "ar.basher@alumni.ubc.ca"
__status__ = "Production"
__description__ = "This file is the main entry to perform learning and prediction on dataset using triUMPF model."

import datetime
import json
import os
import textwrap
from argparse import ArgumentParser

import utility.file_path as fph
from train import train
from utility.arguments import Arguments


def __print_header():
    os.system('clear')
    print('# ' + '=' * 50)
    print('Author: ' + __author__)
    print('Copyright: ' + __copyright__)
    print('License: ' + __license__)
    print('Version: ' + __version__)
    print('Maintainer: ' + __maintainer__)
    print('Email: ' + __email__)
    print('Status: ' + __status__)
    print('Date: ' + datetime.datetime.strptime(__date__, "%d/%m/%Y").strftime("%d-%Q-%Y"))
    print('Description: ' + textwrap.TextWrapper(width=45, subsequent_indent='\t     ').fill(__description__))
    print('# ' + '=' * 50)


def save_args(args):
    os.makedirs(args.log_dir, exist_ok=args.exist_ok)
    with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)


def __internal_args(parse_args):
    arg = Arguments()

    arg.display_interval = parse_args.display_interval
    if parse_args.display_interval < 0:
        arg.display_interval = 1
    arg.random_state = parse_args.random_state
    arg.num_jobs = parse_args.num_jobs
    arg.batch = parse_args.batch
    arg.max_inner_iter = parse_args.max_inner_iter
    arg.num_epochs = parse_args.num_epochs

    ##########################################################################################################
    ##########                                  ARGUMENTS FOR PATHS                                 ##########
    ##########################################################################################################

    arg.ospath = parse_args.ospath
    arg.dspath = parse_args.dspath
    arg.dsfolder = parse_args.dsfolder
    arg.mdpath = parse_args.mdpath
    arg.rspath = parse_args.rspath
    arg.rsfolder = parse_args.rsfolder
    arg.logpath = parse_args.logpath

    ##########################################################################################################
    ##########                          ARGUMENTS FOR FILE NAMES AND MODELS                         ##########
    ##########################################################################################################

    arg.object_name = parse_args.object_name
    arg.pathway2ec_name = parse_args.pathway2ec_name
    arg.pathway2ec_idx_name = parse_args.pathway2ec_idx_name
    arg.features_name = parse_args.features_name
    arg.hin_name = parse_args.hin_name
    arg.M_name = parse_args.M_name
    arg.W_name = parse_args.W_name
    arg.H_name = parse_args.H_name
    arg.X_name = parse_args.X_name
    arg.y_name = parse_args.y_name
    arg.P_name = parse_args.P_name
    arg.E_name = parse_args.E_name
    arg.A_name = parse_args.A_name
    arg.B_name = parse_args.B_name
    arg.file_name = parse_args.file_name
    arg.samples_ids = parse_args.samples_ids
    arg.model_name = parse_args.model_name

    ##########################################################################################################
    ##########                            ARGUMENTS PREPROCESSING FILES                             ##########
    ##########################################################################################################

    arg.preprocess_dataset = parse_args.preprocess_dataset
    arg.white_links = parse_args.white_links

    ##########################################################################################################
    ##########                              ARGUMENTS USED FOR TRAINING                             ##########
    ##########################################################################################################

    arg.train = parse_args.train
    arg.predict = parse_args.predict
    arg.pathway_report = parse_args.pathway_report
    arg.extract_pf = True
    if parse_args.no_parse:
        arg.extract_pf = False
    arg.build_features = True
    if parse_args.no_build_features:
        arg.build_features = False
    arg.plot = parse_args.plot
    arg.num_components = parse_args.num_components
    arg.num_communities_p = parse_args.num_communities_p
    arg.num_communities_e = parse_args.num_communities_e
    arg.no_decomposition = parse_args.no_decomposition
    arg.fit_features = parse_args.fit_features
    arg.proxy_order_p = parse_args.proxy_order_p
    arg.proxy_order_e = parse_args.proxy_order_e
    arg.mu_omega = parse_args.mu_omega
    arg.mu_gamma = parse_args.mu_gamma
    arg.fit_comm = parse_args.fit_comm
    arg.alpha = parse_args.alpha
    arg.beta = parse_args.beta
    arg.rho = parse_args.rho
    arg.lambdas = parse_args.lambdas
    arg.binarize_input_feature = parse_args.binarize
    arg.normalize_input_feature = parse_args.normalize
    arg.use_external_features = parse_args.use_external_features
    arg.cutting_point = parse_args.cutting_point
    arg.fit_intercept = parse_args.fit_intercept
    arg.penalty = parse_args.penalty
    arg.alpha_elastic = parse_args.alpha_elastic
    arg.l1_ratio = 1 - parse_args.l2_ratio
    arg.eps = parse_args.eps
    arg.early_stop = parse_args.early_stop
    arg.loss_threshold = parse_args.loss_threshold
    arg.decision_threshold = parse_args.decision_threshold
    arg.ssample_input_size = parse_args.ssample_input_size
    arg.ssample_label_size = parse_args.ssample_label_size
    arg.top_k = parse_args.top_k
    arg.learning_type = parse_args.learning_type
    arg.lr = parse_args.lr
    arg.lr0 = parse_args.lr0
    arg.forgetting_rate = parse_args.fr
    arg.delay_factor = parse_args.delay
    arg.estimate_prob = parse_args.estimate_prob
    arg.apply_tcriterion = parse_args.apply_tcriterion
    arg.adaptive_beta = parse_args.adaptive_beta
    arg.shuffle = parse_args.shuffle
    return arg


def parse_command_line():
    __print_header()
    # Parses the arguments.
    parser = ArgumentParser(description="Run triUMPF.")

    parser.add_argument('--display-interval', default=2, type=int,
                        help='display intervals. -1 means display per each iteration.')
    parser.add_argument('--random_state', default=12345, type=int, help='Random seed. (default value: 12345).')
    parser.add_argument('--num-jobs', type=int, default=1, help='Number of parallel workers. (default value: 1).')
    parser.add_argument('--batch', type=int, default=30, help='Batch size. (default value: 30).')
    parser.add_argument('--max-inner-iter', default=5, type=int,
                        help='Number of inner iteration. 5. (default value: 5)')
    parser.add_argument('--num-epochs', default=10, type=int,
                        help='Number of epochs over the training set. (default value: 10).')

    # Arguments for path
    parser.add_argument('--ospath', default=fph.OBJECT_PATH, type=str,
                        help='The path to the data object that contains extracted '
                             'information from the MetaCyc database. The default is '
                             'set to object folder outside the source code.')
    parser.add_argument('--dspath', default=fph.DATASET_PATH, type=str,
                        help='The path to the dataset after the samples are processed. '
                             'The default is set to dataset folder outside the source code.')
    parser.add_argument('--dsfolder', default="SAG", type=str,
                        help='The dataset folder name. The default is set to SAG.')
    parser.add_argument('--mdpath', default=fph.MODEL_PATH, type=str,
                        help='The path to the output models. The default is set to '
                             'train folder outside the source code.')
    parser.add_argument('--rspath', default=fph.RESULT_PATH, type=str,
                        help='The path to the results. The default is set to result '
                             'folder outside the source code.')
    parser.add_argument('--rsfolder', default="Prediction_triUMPF", type=str,
                        help='The result folder name. The default is set to Prediction_triUMPF.')
    parser.add_argument('--logpath', default=fph.LOG_PATH, type=str,
                        help='The path to the log directory.')

    # Arguments for file names and models
    parser.add_argument('--object-name', type=str, default='biocyc.pkl',
                        help='The biocyc file name. (default value: "biocyc.pkl")')
    parser.add_argument('--pathway2ec-name', type=str, default='pathway2ec.pkl',
                        help='The pathway2ec association matrix file name. (default value: "pathway2ec.pkl")')
    parser.add_argument('--pathway2ec-idx-name', type=str, default='pathway2ec_idx.pkl',
                        help='The pathway2ec association indices file name. (default value: "pathway2ec_idx.pkl")')
    parser.add_argument('--features-name', type=str, default='path2vec_cmt_tf_embeddings.npz',
                        help='The features file name. (default value: "path2vec_cmt_tf_embeddings.npz")')
    parser.add_argument('--hin-name', type=str, default='hin_cmt.pkl',
                        help='The hin file name. (default value: "hin_cmt.pkl")')
    parser.add_argument('--M-name', type=str, default='M.pkl',
                        help='The pathway2ec association matrix file name. (default value: "M.pkl")')
    parser.add_argument('--W-name', type=str, default='W.pkl',
                        help='The W file name. (default value: "W.pkl")')
    parser.add_argument('--H-name', type=str, default='H.pkl',
                        help='The H file name. (default value: "H.pkl")')
    parser.add_argument('--X-name', type=str, default='golden_Xe.pkl',
                        help='The X file name. (default value: "biocyc_Xe.pkl")')
    parser.add_argument('--y-name', type=str, default='golden_y.pkl',
                        help='The X file name. (default value: "biocyc_y.pkl")')
    parser.add_argument('--P-name', type=str, default='P.pkl',
                        help='The pathway features file name. (default value: "P.pkl")')
    parser.add_argument('--E-name', type=str, default='E.pkl',
                        help='The EC features file name. (default value: "E.pkl")')
    parser.add_argument('--A-name', type=str, default='A.pkl',
                        help='The pathway to pathway association file name. (default value: "A.pkl")')
    parser.add_argument('--B-name', type=str, default='B.pkl',
                        help='The EC to EC association file name. (default value: "B.pkl")')
    parser.add_argument('--samples-ids', type=str, default='SAG_ids.pkl',
                        help='The samples ids file name. (default value: "SAG_ids.pkl")')
    parser.add_argument('--file-name', type=str, default='triUMPF_symbionts',
                        help='The file name to save various scores and communities files. (default value: "triUMPF_O_final")')
    parser.add_argument('--model-name', type=str, default='triUMPF_C_final',
                        help='The file name, excluding extension, to save an object. (default value: "triUMPF_O_final")')

    # Arguments for preprocessing dataset
    parser.add_argument('--preprocess-dataset', action='store_true', default=False,
                        help='Preprocess dataset. (default value: False).')
    parser.add_argument('--white-links', action='store_true', default=False,
                        help='Add no noise to Pathway-Pathway and EC-EC associations. (default value: False).')

    # Arguments for training and evaluation
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train the triUMPF model. (default value: False).')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='Whether to predict labels from inputs. (default value: False).')
    parser.add_argument('--pathway-report', action='store_true', default=False,
                        help='Whether to generate a detailed report for pathways for each instance. '
                             '(default value: False).')
    parser.add_argument('--no-parse', action='store_true', default=False,
                        help='Whether to parse Pathologic format file (pf) from a folder (default value: False).')
    parser.add_argument('--no-build-features', action='store_true', default=True,
                        help='Whether to construct features (default value: True).')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Whether to produce various plots from predicted outputs. '
                             '(default value: False).')
    parser.add_argument('--num-components', default=100, type=int,
                        help='Number of components. (default value: 100).')
    parser.add_argument('--num-communities-p', default=90, type=int,
                        help='Number of communities for pathways. (default value: 90).')
    parser.add_argument('--num-communities-e', default=100, type=int,
                        help='Number of communities for ecs. (default value: 100).')
    parser.add_argument('--proxy-order-p', type=int, default=3,
                        help='Higher order proxy for pathway-pathway adjacency. (default value: 1).')
    parser.add_argument('--proxy-order-e', type=int, default=1,
                        help='Higher order proxy for EC-EC adjacency. (default value: 1).')
    parser.add_argument('--mu-omega', type=float, default=0.1,
                        help='Weight for the higher order proxy for pathway-pathway adjacency. (default value: 0.3).')
    parser.add_argument('--mu-gamma', type=float, default=0.3,
                        help='Weight for the higher order proxy for EC-EC adjacency. (default value: 0.3).')
    parser.add_argument('--no-decomposition', action='store_true', default=False,
                        help='Whether to decompose pathway-EC association matrix. (default value: False).')
    parser.add_argument('--fit-features', action='store_true', default=False,
                        help='Whether to fit by external features. (default value: False).')
    parser.add_argument('--fit-comm', action='store_true', default=False,
                        help='Whether to fit community. (default value: False).')
    parser.add_argument('--binarize', action='store_true', default=False,
                        help='Whether to binarize data (set feature values to 0 or 1). (default value: False).')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='Whether to normalize data. (default value: False).')
    parser.add_argument('--use-external-features', action='store_true', default=False,
                        help='Whether to use external features that are included in data. '
                             '(default value: False).')
    parser.add_argument('--cutting-point', type=int, default=3650,
                        help='The cutting point after which binarize operation is halted in data. '
                             '(default value: 3650).')
    parser.add_argument('--fit-intercept', action='store_false', default=True,
                        help='Whether the intercept should be estimated or not. (default value: True).')
    parser.add_argument('--penalty', default='l21', type=str, choices=['l1', 'l2', 'elasticnet', 'l21', 'none'],
                        help='The penalty (aka regularization term) to be used. (default value: "none")')
    parser.add_argument('--alpha-elastic', default=0.0001, type=float,
                        help='Constant that multiplies the regularization term to control '
                             'the amount to regularize parameters and in our paper it is lambda. '
                             '(default value: 0.0001)')
    parser.add_argument('--l2-ratio', default=0.35, type=float,
                        help='The elastic net mixing parameter, with 0 <= l2_ratio <= 1. l2_ratio=0 '
                             'corresponds to L1 penalty, l2_ratio=1 to L2. (default value: 0.35)')
    parser.add_argument("--alpha", type=float, default=1e9,
                        help="A hyper-parameter to satisfy orthogonal condition. (default value: 1e9).")
    parser.add_argument("--beta", type=float, default=1e9,
                        help="A hyper-parameter to satisfy orthogonal condition. (default value: 1e9).")
    parser.add_argument("--rho", type=float, default=0.01,
                        help="A hyper-parameter to fuse coefficients with association matrix. (default value: 0.01).")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                        help="Six hyper-parameters for constraints. Default is [0.01, 0.01, 7, 0.01, 0.01, 0.01].")
    parser.add_argument('--eps', default=1e-4, type=float,
                        help='Truncate all values less then this in output to zero. (default value: 1e-4).')
    parser.add_argument("--early-stop", action='store_true', default=False,
                        help="Whether to terminate training based on relative change "
                             "between two consecutive iterations. (default value: False).")
    parser.add_argument("--loss-threshold", type=float, default=0.005,
                        help="A hyper-parameter for deciding the cutoff threshold of the differences "
                             "of loss between two consecutive rounds. (default value: 0.005).")
    parser.add_argument("--decision-threshold", type=float, default=0.5,
                        help="The cutoff threshold for triUMPF. (default value: 0.5)")
    parser.add_argument('--top-k', type=int, default=10,
                        help='Top k features. (default value: 10).')
    parser.add_argument('--ssample-input-size', default=0.05, type=float,
                        help='The size of input subsample. (default value: 0.05)')
    parser.add_argument('--ssample-label-size', default=50, type=int,
                        help='Maximum number of labels to be sampled. (default value: 50).')
    parser.add_argument('--learning-type', default='optimal', type=str, choices=['optimal', 'sgd'],
                        help='The learning rate schedule. (default value: "optimal")')
    parser.add_argument('--lr', default=0.0001, type=float, help='The learning rate. (default value: 0.0001).')
    parser.add_argument('--lr0', default=0.0, type=float, help='The initial learning rate. (default value: 0.0).')
    parser.add_argument('--fr', type=float, default=0.9,
                        help='Forgetting rate to control how quickly old information is forgotten. The value should '
                             'be set between (0.5, 1.0] to guarantee asymptotic convergence. (default value: 0.7).')
    parser.add_argument('--delay', type=float, default=1.,
                        help='Delay factor down weights early iterations. (default value: 1).')
    parser.add_argument('--estimate-prob', action='store_true', default=False,
                        help='Whether to return prediction of labels and bags as probability '
                             'estimate or not. (default value: False).')
    parser.add_argument('--apply-tcriterion', action='store_true', default=False,
                        help='Whether to employ adaptive strategy during prediction. (default value: False).')
    parser.add_argument('--adaptive-beta', default=0.45, type=float,
                        help='The adaptive beta parameter for prediction. (default value: 0.45).')
    parser.add_argument('--shuffle', action='store_false', default=True,
                        help='Whether or not the training data should be shuffled after each epoch. '
                             '(default value: True).')

    parse_args = parser.parse_args()
    args = __internal_args(parse_args)

    train(arg=args)


if __name__ == "__main__":
    # app.run(parse_command_line)
    parse_command_line()
