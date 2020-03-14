'''
This file is the main entry used to train the input dataset
using triUMPF train and also report the predicted vocab.
'''

import os
import sys
import time
import traceback

import networkx as nx
import numpy as np
from model.triumpf import triUMPF
from scipy.sparse import lil_matrix, hstack
from sklearn import preprocessing
from sklearn.utils._joblib import Parallel, delayed
from utility.access_file import load_data, save_data
from utility.model_utils import score, synthesize_report, compute_abd_cov
from utility.parse_input import parse_files


def __build_features(X, pathwat_dict, ec_dict, labels_components, node2idx_pathway2ec, path2vec_features, file_name,
                     dspath, batch_size=100, num_jobs=1):
    tmp = lil_matrix.copy(X)
    print('\t>> Build abundance and coverage features...')
    list_batches = np.arange(start=0, stop=tmp.shape[0], step=batch_size)
    total_progress = len(list_batches) * len(pathwat_dict.keys())
    parallel = Parallel(n_jobs=num_jobs, verbose=0)
    results = parallel(delayed(compute_abd_cov)(tmp[batch:batch + batch_size],
                                                labels_components, pathwat_dict,
                                                None, batch_idx, total_progress)
                       for batch_idx, batch in enumerate(list_batches))
    desc = '\t\t--> Building {0:.4f}%...'.format((100))
    print(desc)
    abd, cov = zip(*results)
    abd = np.vstack(abd)
    cov = np.vstack(cov)
    del results
    abd = preprocessing.normalize(abd)
    print('\t>> Use pathway2vec EC features...')
    path2vec_features = path2vec_features[path2vec_features.files[0]]
    path2vec_features = path2vec_features / np.linalg.norm(path2vec_features, axis=1)[:, np.newaxis]
    ec_features = [idx for idx, v in ec_dict.items() if v in node2idx_pathway2ec]
    path2vec_features = path2vec_features[ec_features, :]
    ec_features = [np.mean(path2vec_features[row.rows[0]] * np.array(row.data[0])[:, None], axis=0)
                   for idx, row in enumerate(X)]
    save_data(data=lil_matrix(ec_features), file_name=file_name + "_Xp.pkl", save_path=dspath, mode="wb",
              tag="transformed instances to ec features")
    X = lil_matrix(hstack((tmp, ec_features)))
    save_data(data=X, file_name=file_name + "_Xe.pkl", save_path=dspath, mode="wb",
              tag="concatenated ec features with instances")
    X = lil_matrix(hstack((tmp, abd)))
    save_data(data=X, file_name=file_name + "_Xa.pkl", save_path=dspath, mode="wb",
              tag="concatenated abundance features with instances")
    X = lil_matrix(hstack((tmp, cov)))
    save_data(data=X, file_name=file_name + "_Xc.pkl", save_path=dspath, mode="wb",
              tag="concatenated coverage features with instances")
    X = lil_matrix(hstack((tmp, ec_features)))
    X = lil_matrix(hstack((X, abd)))
    save_data(data=X, file_name=file_name + "_Xea.pkl", save_path=dspath, mode="wb",
              tag="concatenated ec and abundance features with instances")
    X = lil_matrix(hstack((tmp, ec_features)))
    X = lil_matrix(hstack((X, cov)))
    save_data(data=X, file_name=file_name + "_Xec.pkl", save_path=dspath, mode="wb",
              tag="concatenated ec and coverage features with instances")
    X = lil_matrix(hstack((tmp, ec_features)))
    X = lil_matrix(hstack((X, abd)))
    X = lil_matrix(hstack((X, cov)))
    save_data(data=X, file_name=file_name + "_Xm.pkl", save_path=dspath, mode="wb",
              tag="concatenated ec, abundance, and coverage features features with instances")


###***************************        Private Main Entry        ***************************###

def __train(arg):
    # Setup the number of operations to employ
    steps = 1
    # Whether to display parameters at every operation
    display_params = True

    if arg.preprocess_dataset:
        print('\n{0})- Preprocessing dataset...'.format(steps))
        steps = steps + 1

        print('\t>> Loading files...')
        # load a biocyc file
        data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object')
        # extract pathway ids
        pathway_dict = data_object["pathway_id"]
        ec_dict = data_object["ec_id"]
        del data_object

        # load a hin file
        hin = load_data(file_name=arg.hin_name, load_path=arg.ospath,
                        tag='heterogeneous information network')
        # get path2vec mapping
        node2idx_path2vec = dict((node[0], node[1]['mapped_idx'])
                                 for node in hin.nodes(data=True))
        # get pathway2ec mapping
        node2idx_pathway2ec = [node[0] for node in hin.nodes(data=True)]
        Adj = nx.adj_matrix(G=hin)
        del hin

        # load pathway2ec mapping matrix
        pathway2ec_idx = load_data(file_name=arg.pathway2ec_idx_name, load_path=arg.ospath)
        path2vec_features = np.load(file=os.path.join(arg.mdpath, arg.features_name))

        # extracting pathway and ec features
        labels_components = load_data(file_name=arg.pathway2ec_name, load_path=arg.ospath, tag='M')
        path2vec_features = path2vec_features[path2vec_features.files[0]]
        pathways_idx = np.array([node2idx_path2vec[v] for v, idx in pathway_dict.items()
                                 if v in node2idx_path2vec])
        P = path2vec_features[pathways_idx, :]
        tmp = [idx for v, idx in ec_dict.items() if v in node2idx_pathway2ec]
        ec_idx = np.array([idx for idx in tmp if len(np.where(pathway2ec_idx == idx)[0]) > 0])
        E = path2vec_features[ec_idx, :]

        # constraint features space between 0 to 1 to avoid negative results
        min_rho = np.min(P)
        max_rho = np.max(P)
        P = P - min_rho
        P = P / (max_rho - min_rho)
        P = P / np.linalg.norm(P, axis=1)[:, np.newaxis]
        min_rho = np.min(E)
        max_rho = np.max(E)
        E = E - min_rho
        E = E / (max_rho - min_rho)
        E = E / np.linalg.norm(E, axis=1)[:, np.newaxis]

        # building A and B matrices
        lil_matrix.setdiag(Adj, 0)
        A = Adj[pathways_idx[:, None], pathways_idx]
        A = A / A.sum(1)
        A = np.nan_to_num(A)
        B = Adj[ec_idx[:, None], ec_idx]
        B = B / B.sum(1)
        B = np.nan_to_num(B)

        ## train size
        if arg.ssample_input_size < 1:
            # add white noise to M
            train_size = labels_components.shape[0] * arg.ssample_input_size
            idx = np.random.choice(a=np.arange(labels_components.shape[0]), size=int(train_size), replace=False)
            labels_components = labels_components.toarray()
            labels_components[idx] = np.zeros((idx.shape[0], labels_components.shape[1]))
        if arg.white_links:
            if arg.ssample_input_size < 1:
                # add white noise to A
                train_size = A.shape[0] * arg.ssample_input_size
                idx = np.random.choice(a=np.arange(A.shape[0]), size=int(train_size), replace=False)
                A = lil_matrix(A).toarray()
                tmp = np.zeros((idx.shape[0], A.shape[0]))
                A[idx] = tmp
                A[:, idx] = tmp.T
                # add white noise to B
                train_size = B.shape[0] * arg.ssample_input_size
                idx = np.random.choice(a=np.arange(B.shape[0]), size=int(train_size), replace=False)
                B = lil_matrix(B).toarray()
                tmp = np.zeros((idx.shape[0], B.shape[0]))
                B[idx] = tmp
                B[:, idx] = tmp.T

        # save files
        print('\t>> Saving files...')
        save_data(data=lil_matrix(labels_components), file_name=arg.M_name, save_path=arg.dspath, tag="M", mode="wb")
        save_data(data=lil_matrix(P), file_name=arg.P_name, save_path=arg.dspath, tag="P", mode="wb")
        save_data(data=lil_matrix(E), file_name=arg.E_name, save_path=arg.dspath, tag="E", mode="wb")
        save_data(data=lil_matrix(A), file_name=arg.A_name, save_path=arg.dspath, tag="A", mode="wb")
        save_data(data=lil_matrix(B), file_name=arg.B_name, save_path=arg.dspath, tag="B", mode="wb")
        print('\t>> Done...')

    ##########################################################################################################
    ######################                     TRAIN USING triUMPF                      ######################
    ##########################################################################################################

    if arg.train:
        print('\n{0})- Training {1} dataset using triUMPF model...'.format(steps, arg.y_name))
        steps = steps + 1

        # load files
        print('\t>> Loading files...')
        labels_components, W, H, P, E, A, B, X, y = None, None, None, None, None, None, None, None, None

        if arg.no_decomposition:
            W = load_data(file_name=arg.W_name, load_path=arg.mdpath, tag='W')
            H = load_data(file_name=arg.H_name, load_path=arg.mdpath, tag='H')
        else:
            labels_components = load_data(file_name=arg.M_name, load_path=arg.dspath, tag='M')
        if arg.fit_features:
            P = load_data(file_name=arg.P_name, load_path=arg.dspath, tag='P')
            E = load_data(file_name=arg.E_name, load_path=arg.dspath, tag='E')
        if arg.fit_comm:
            if not arg.fit_features:
                P = load_data(file_name=arg.P_name, load_path=arg.dspath, tag='P')
                E = load_data(file_name=arg.E_name, load_path=arg.dspath, tag='E')
            X = load_data(file_name=arg.X_name, load_path=arg.dspath, tag='X')
            y = load_data(file_name=arg.y_name, load_path=arg.dspath, tag='X')
            A = load_data(file_name=arg.A_name, load_path=arg.dspath, tag='A')
            B = load_data(file_name=arg.B_name, load_path=arg.dspath, tag='B')

        model = triUMPF(num_components=arg.num_components, num_communities_p=arg.num_communities_p,
                        num_communities_e=arg.num_communities_e, proxy_order_p=arg.proxy_order_p,
                        proxy_order_e=arg.proxy_order_e, mu_omega=arg.mu_omega, mu_gamma=arg.mu_gamma,
                        fit_features=arg.fit_features, fit_comm=arg.fit_comm,
                        normalize_input_feature=arg.normalize_input_feature,
                        binarize_input_feature=arg.binarize_input_feature,
                        use_external_features=arg.use_external_features, cutting_point=arg.cutting_point,
                        fit_intercept=arg.fit_intercept, alpha=arg.alpha, beta=arg.beta, rho=arg.rho,
                        lambdas=arg.lambdas, eps=arg.eps, early_stop=arg.early_stop, penalty=arg.penalty,
                        alpha_elastic=arg.alpha_elastic, l1_ratio=arg.l1_ratio, loss_threshold=arg.loss_threshold,
                        decision_threshold=arg.decision_threshold, subsample_input_size=arg.ssample_input_size,
                        subsample_labels_size=arg.ssample_label_size, learning_type=arg.learning_type, lr=arg.lr,
                        lr0=arg.lr0, delay_factor=arg.delay_factor, forgetting_rate=arg.forgetting_rate,
                        batch=arg.batch, max_inner_iter=arg.max_inner_iter, num_epochs=arg.num_epochs,
                        num_jobs=arg.num_jobs, display_interval=arg.display_interval, shuffle=arg.shuffle,
                        random_state=arg.random_state, log_path=arg.logpath)
        model.fit(M=labels_components, W=W, H=H, X=X, y=y, P=P, E=E, A=A, B=B, model_name=arg.model_name,
                  model_path=arg.mdpath, result_path=arg.rspath, display_params=display_params)

    ##########################################################################################################
    ######################                   EVALUATE USING triUMPF                     ######################
    ##########################################################################################################

    if arg.evaluate:
        print('\n{0})- Evaluating triUMPF model...'.format(steps))
        steps = steps + 1

        # load files
        print('\t>> Loading files...')
        labels_components = load_data(file_name=arg.M_name, load_path=arg.dspath, tag='M')
        W = load_data(file_name=arg.W_name, load_path=arg.mdpath, tag='W')
        H = load_data(file_name=arg.H_name, load_path=arg.mdpath, tag='H')

        # load model
        model = load_data(file_name=arg.model_name + '.pkl', load_path=arg.mdpath, tag='triUMPF model')

        P, E, A, B, X, y = None, None, None, None, None, None

        if model.fit_features:
            P = load_data(file_name=arg.P_name, load_path=arg.dspath, tag='P')
            E = load_data(file_name=arg.E_name, load_path=arg.dspath, tag='E')
        if model.fit_comm:
            if not model.fit_features:
                P = load_data(file_name=arg.P_name, load_path=arg.dspath, tag='P')
                E = load_data(file_name=arg.E_name, load_path=arg.dspath, tag='E')
            X = load_data(file_name=arg.X_name, load_path=arg.dspath, tag='X')
            y = load_data(file_name=arg.y_name, load_path=arg.dspath, tag='X')
            A = load_data(file_name=arg.A_name, load_path=arg.dspath, tag='A')
            B = load_data(file_name=arg.B_name, load_path=arg.dspath, tag='B')
            C = load_data(file_name=arg.model_name + "_C.pkl", load_path=arg.mdpath, tag='C')
            K = load_data(file_name=arg.model_name + "_K.pkl", load_path=arg.mdpath, tag='K')
            T = load_data(file_name=arg.model_name + "_T.pkl", load_path=arg.mdpath, tag='T')
            R = load_data(file_name=arg.model_name + "_R.pkl", load_path=arg.mdpath, tag='R')

        # load a biocyc file
        data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object')
        pathway_dict = data_object["pathway_id"]
        ec_dict = data_object["ec_id"]
        del data_object
        pathway_dict = dict((idx, id) for id, idx in pathway_dict.items())
        ec_dict = dict((idx, id) for id, idx in ec_dict.items())

        # reconstruction error
        M_hat = model.inverse_transform(X1=W.toarray(), X2=H.toarray())
        err = model.rec_err(M=labels_components.toarray(), M_hat=M_hat.toarray())
        print("\t>> Reconstruction error of the association matrix: {0:.4f}".format(err))
        file_name = arg.file_name + '_rec_error.txt'
        save_data(data="# Reconstruction error of the association matrix: {0:.4f}\n".format(err),
                  file_name=file_name, save_path=arg.rspath, tag="reconstruction error",
                  mode='w', w_string=True, print_tag=True)
        if model.fit_comm:
            A_hat = model.inverse_transform(X1=P.toarray(), X2=C.toarray(), X3=T.toarray(), comm=True)
            err = model.rec_err(M=A.toarray(), M_hat=A_hat.toarray())
            print("\t>> Reconstruction error of label community: {0:.4f}".format(err))
            save_data(data="# Reconstruction error of label community: {0:.4f}\n".format(err),
                      file_name=file_name, save_path=arg.rspath, tag="reconstruction error",
                      mode='a', w_string=True, print_tag=False)

            B_hat = model.inverse_transform(X1=E.toarray(), X2=K.toarray(), X3=R.toarray(), comm=True)
            err = model.rec_err(M=B.toarray(), M_hat=B_hat.toarray())
            print("\t>> Reconstruction error of features community: {0:.4f}".format(err))
            save_data(data="# Reconstruction error of features community: {0:.4f}".format(err),
                      file_name=file_name, save_path=arg.rspath, tag="reconstruction error",
                      mode='a', w_string=True, print_tag=False)

            # labels prediction score
            y_pred = model.predict(X=X.toarray(), estimate_prob=arg.estimate_prob,
                                   apply_t_criterion=arg.apply_tcriterion,
                                   adaptive_beta=arg.adaptive_beta, decision_threshold=arg.decision_threshold,
                                   top_k=arg.top_k, batch_size=arg.batch, num_jobs=arg.num_jobs)
            file_name = arg.file_name + '_scores.txt'
            score(y_true=y.toarray(), y_pred=y_pred.toarray(), item_lst=['biocyc'], six_db=False, mode='a',
                  file_name=file_name, save_path=arg.rspath)
            score(y_true=y.toarray(), y_pred=y_pred.toarray(), item_lst=['biocyc'], six_db=True, mode='a',
                  file_name=file_name, save_path=arg.rspath)
            # top features per cluster
            clusters_dict = model.get_top_features(X=H.toarray(), feature_names=ec_dict, top_k_features=arg.top_k)
            print("\t>> Clusters with top {0} features...".format(arg.top_k))
            file_name = arg.file_name + '_clusters.txt'
            save_data(data="# Clusters with top {0} features...\n".format(arg.top_k),
                      file_name=file_name, save_path=arg.rspath,
                      tag="clusters", mode='w', w_string=True, print_tag=True)
            for cluster_idx, features in clusters_dict.items():
                save_data(data="  >> Cluster {0}: {1}\n".format(cluster_idx, ', '.join(str(feat) for feat in features)),
                          file_name=file_name, save_path=arg.rspath, mode='a',
                          w_string=True, print_tag=False)

        if model.fit_comm:
            # top features per communty
            communities_dict = model.get_top_features(X=C.toarray(), feature_names=pathway_dict,
                                                      top_k_features=arg.top_k)
            print("\t>> Communities with top {0} features...".format(arg.top_k))

            file_name = arg.file_name + '_pathway_communities.txt'
            save_data(data="# Communities with top {0} features...\n".format(arg.top_k),
                      file_name=file_name, save_path=arg.rspath, tag="communities", mode='w',
                      w_string=True, print_tag=True)
            for community_idx, features in communities_dict.items():
                save_data(
                    data="  >> Community {0}: {1}\n".format(community_idx, ', '.join(str(feat) for feat in features)),
                    file_name=file_name, save_path=arg.rspath, mode='a', w_string=True,
                    print_tag=False)

            communities_dict = model.get_top_features(X=K.toarray(), feature_names=ec_dict, top_k_features=arg.top_k)
            file_name = arg.file_name + '_ec_communities.txt'
            save_data(data="\n# Communities with top {0} features...\n".format(arg.top_k),
                      file_name=file_name, save_path=arg.rspath, mode='a', w_string=True, print_tag=False)
            for community_idx, features in communities_dict.items():
                save_data(
                    data="  >> Community {0}: {1}\n".format(community_idx, ', '.join(str(feat) for feat in features)),
                    file_name=file_name, save_path=arg.rspath, mode='a', w_string=True, print_tag=False)

    ##########################################################################################################
    ######################                    PREDICT USING triUMPF                     ######################
    ##########################################################################################################

    if arg.predict:
        print('\n{0})- Predicting using a pre-trained triUMPF model...'.format(steps))
        if arg.pathway_report:
            print('\t>> Loading biocyc object...')
            # load a biocyc file
            data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object',
                                    print_tag=False)
            pathway_dict = data_object["pathway_id"]
            pathway_common_names = dict((pidx, data_object['processed_kb']['metacyc'][5][pid][0][1])
                                        for pid, pidx in pathway_dict.items()
                                        if pid in data_object['processed_kb']['metacyc'][5])
            ec_dict = data_object['ec_id']
            del data_object
            pathway_dict = dict((idx, id) for id, idx in pathway_dict.items())
            ec_dict = dict((idx, id) for id, idx in ec_dict.items())
            labels_components = load_data(file_name=arg.pathway2ec_name, load_path=arg.ospath, tag='M')
            print('\t>> Loading label to component mapping file object...')
            pathway2ec_idx = load_data(file_name=arg.pathway2ec_idx_name, load_path=arg.ospath, print_tag=False)
            pathway2ec_idx = list(pathway2ec_idx)
            tmp = list(ec_dict.keys())
            ec_dict = dict((idx, ec_dict[tmp.index(ec)]) for idx, ec in enumerate(pathway2ec_idx))
            if arg.extract_pf:
                X, sample_ids = parse_files(ec_dict=ec_dict, input_folder=arg.dsfolder, rsfolder=arg.rsfolder,
                                            rspath=arg.rspath, num_jobs=arg.num_jobs)
                print('\t>> Storing X and sample_ids...')
                save_data(data=X, file_name=arg.file_name + '_X.pkl', save_path=arg.dspath,
                          tag='the pf dataset (X)', mode='w+b', print_tag=False)
                save_data(data=sample_ids, file_name=arg.file_name + '_ids.pkl', save_path=arg.dspath,
                          tag='samples ids', mode='w+b', print_tag=False)
                if arg.build_features:
                    # load a hin file
                    print('\t>> Loading heterogeneous information network file...')
                    hin = load_data(file_name=arg.hin_name, load_path=arg.ospath,
                                    tag='heterogeneous information network',
                                    print_tag=False)
                    # get pathway2ec mapping
                    node2idx_pathway2ec = [node[0] for node in hin.nodes(data=True)]
                    del hin
                    print('\t>> Loading path2vec_features file...')
                    path2vec_features = np.load(file=os.path.join(arg.mdpath, arg.features_name))
                    __build_features(X=X, pathwat_dict=pathway_dict, ec_dict=ec_dict,
                                     labels_components=labels_components,
                                     node2idx_pathway2ec=node2idx_pathway2ec,
                                     path2vec_features=path2vec_features,
                                     file_name=arg.file_name, dspath=arg.dspath,
                                     batch_size=arg.batch, num_jobs=arg.num_jobs)
        # load files
        print('\t>> Loading necessary files......')
        X = load_data(file_name=arg.X_name, load_path=arg.dspath, tag="X")
        sample_ids = np.arange(X.shape[0])
        if arg.samples_ids in os.listdir(arg.dspath):
            sample_ids = load_data(file_name=arg.samples_ids, load_path=arg.dspath, tag="samples ids")

        # load model
        model = load_data(file_name=arg.model_name + '.pkl', load_path=arg.mdpath, tag='triUMPF model')

        # predict
        y_pred = model.predict(X=X.toarray(), estimate_prob=False, apply_t_criterion=arg.apply_tcriterion,
                               adaptive_beta=arg.adaptive_beta, decision_threshold=arg.decision_threshold,
                               top_k=arg.top_k, batch_size=arg.batch, num_jobs=arg.num_jobs)
        # labels prediction score
        y_pred_score = model.predict(X=X.toarray(), estimate_prob=True, apply_t_criterion=arg.apply_tcriterion,
                                     adaptive_beta=arg.adaptive_beta, decision_threshold=arg.decision_threshold,
                                     top_k=arg.top_k, batch_size=arg.batch, num_jobs=arg.num_jobs)

        if arg.pathway_report:
            print('\t>> Synthesizing pathway reports...')
            synthesize_report(X=X[:, :arg.cutting_point], sample_ids=sample_ids,
                              y_pred=y_pred, y_dict_ids=pathway_dict, y_common_name=pathway_common_names,
                              component_dict=ec_dict, labels_components=labels_components, y_pred_score=y_pred_score,
                              batch_size=arg.batch, num_jobs=arg.num_jobs, rsfolder=arg.rsfolder, rspath=arg.rspath,
                              dspath=arg.dspath, file_name=arg.file_name + '_triumpf')
        else:
            print('\t>> Storing predictions (label index) to: {0:s}'.format(arg.file_name + '_triumpf_y.pkl'))
            save_data(data=y_pred, file_name=arg.file_name + "_triumpf_y.pkl", save_path=arg.dspath,
                      mode="wb", print_tag=False)


def train(arg):
    try:
        if arg.preprocess_dataset or arg.train or arg.evaluate or arg.predict:
            actions = list()
            if arg.preprocess_dataset:
                actions += ['PREPROCESS DATASETs']
            if arg.train:
                actions += ['TRAIN MODELs']
            if arg.evaluate:
                actions += ['EVALUATE MODELs']
            if arg.predict:
                actions += ['PREDICT RESULTS USING SPECIFIED MODELs']
            desc = [str(item[0] + 1) + '. ' + item[1] for item in zip(list(range(len(actions))), actions)]
            desc = ' '.join(desc)
            print('\n*** APPLIED ACTIONS ARE: {0}'.format(desc))
            timeref = time.time()
            __train(arg)
            print('\n*** The selected actions consumed {1:f} SECONDS\n'.format('', round(time.time() - timeref, 3)),
                  file=sys.stderr)
        else:
            print('\n*** PLEASE SPECIFY AN ACTION...\n', file=sys.stderr)
    except Exception:
        print(traceback.print_exc())
        raise
