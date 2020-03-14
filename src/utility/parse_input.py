'''
This file preprocesses the input data in PathoLogic File Format (.pf).
'''

import numpy as np
import os
import os.path
import shutil
import sys
from collections import OrderedDict
from scipy.sparse import lil_matrix
from sklearn.utils._joblib import Parallel, delayed


def create_remove_dir(folder_path):
    if os.path.isdir(folder_path):
        try:
            shutil.rmtree(path=folder_path)
        except OSError as e:
            print("\t\t## Cannot remove the directory: {0}".format(folder_path), file=sys.stderr)
            raise e
    try:
        os.mkdir(path=folder_path)
    except OSError as e:
        print("\t\t## Creation of the directory {0} failed...".format(folder_path), file=sys.stderr)
        raise e


def copy_files(destination_path):
    if os.path.isdir(destination_path):
        try:
            shutil.rmtree(path=folder_path)
        except OSError as e:
            print("\t\t## Cannot remove the directory: {0}".format(folder_path), file=sys.stderr)
            raise e
    try:
        os.mkdir(path=folder_path)
    except OSError as e:
        print("\t\t## Creation of the directory {0} failed...".format(folder_path), file=sys.stderr)
        raise e


def __parse_pf_file(input_file):
    if os.path.isfile(input_file):
        product_info = OrderedDict()
        with open(input_file, errors='ignore') as f:
            for text in f:
                if not str(text).startswith('#'):
                    ls = text.strip().split('\t')
                    if ls:
                        if ls[0] == 'ID':
                            product_id = ' '.join(ls[1:])
                            product_name = ''
                            product_type = ''
                            product_ec = ''
                        elif ls[0] == 'PRODUCT':
                            product_name = ' '.join(ls[1:])
                        elif ls[0] == 'PRODUCT-TYPE':
                            product_type = ' '.join(ls[1:])
                        elif ls[0] == 'EC':
                            product_ec = 'EC-'
                            product_ec = product_ec + ''.join(ls[1:])
                        elif ls[0] == '//':
                            # datum is comprised of {ID: (PRODUCT, PRODUCT-TYPE, EC)}
                            datum = {product_id: (product_name, product_type, product_ec)}
                            product_info.update(datum)
        return product_info


def __extract_ecs(input_path):
    """ Process input from a given path
    :type input_path: str
    :param input_path: The RunPathoLogic input path, where all the data folders
        are located
    """
    found_pf = False
    for file_name in os.listdir(input_path):
        if file_name.endswith('.pf'):
            input_file = os.path.join(input_path, file_name)
            found_pf = True
            break
    if found_pf:
        product_info = __parse_pf_file(input_file)
    else:
        pass
    return product_info


def __extract_input_from_pf_files(ipath, idx, total_samples):
    if os.path.isdir(ipath) or os.path.exists(ipath):
        # Preprocess inputs
        input_info = __extract_ecs(input_path=ipath)
        input_results = list()
        for i, item in input_info.items():
            if item[2]:
                input_results.append(item[2])
        desc = '\t   --> Processed: {0:.2f}%'.format(((idx + 1) / total_samples) * 100)
        if idx + 1 != total_samples:
            print(desc, end="\r")
        if idx + 1 == total_samples:
            print(desc)
    else:
        print('\t>> Failed to preprocess {0} file...'.format(ipath.split('/')[-2]),
              file=sys.stderr)
    return input_results


def __preprocess(ec_dict, data_folder_path, result_folder_path, num_jobs=2):
    lst_ipaths = sorted([os.path.join(data_folder_path, folder) for folder in os.listdir(data_folder_path)
                         if os.path.isdir(os.path.join(data_folder_path, folder))])
    if len(lst_ipaths) == 0:
        X, sample_ids = 0, 0
        return X, sample_ids
    print('\t>> Copy {0} files into {1}...'.format(len(lst_ipaths), result_folder_path))
    create_remove_dir(folder_path=result_folder_path)
    sample_ids = [os.path.split(opath)[-1] for opath in lst_ipaths]
    for sidx, sid in enumerate(sample_ids):
        create_remove_dir(folder_path=os.path.join(result_folder_path, sid))
        # input_path = os.path.join(lst_ipaths[sidx], 'ptools')
        input_path = os.path.join(lst_ipaths[sidx], '1.0/input')
        for file_name in os.listdir(input_path):
            if file_name.endswith('.pf'):
                input_file = os.path.join(input_path, file_name)
                shutil.copy(input_file, os.path.join(result_folder_path, sid))
                break
    lst_ipaths = sorted([os.path.join(result_folder_path, folder) for folder in os.listdir(result_folder_path)
                         if os.path.isdir(os.path.join(data_folder_path, folder))])
    print('\t>> Extracting input information from {0} files...'.format(len(lst_ipaths)))
    parallel = Parallel(n_jobs=num_jobs, verbose=0)
    results = parallel(delayed(__extract_input_from_pf_files)(item, idx, len(lst_ipaths))
                       for idx, item in enumerate(lst_ipaths))
    output = results
    X = np.zeros((len(output), len(ec_dict.keys())), dtype=np.int32)
    ec_dict = dict((id, idx) for idx, id in ec_dict.items())
    for list_idx, item in enumerate(output):
        for ec in item:
            if ec in ec_dict:
                X[list_idx, ec_dict[ec]] += 1
        desc = '\t   --> Trimming ECs: {0:.2f}%'.format(((list_idx + 1) / len(output)) * 100)
        if list_idx + 1 != len(output):
            print(desc, end="\r")
        if list_idx + 1 == len(output):
            print(desc)
    list_idx = [idx for idx, item in enumerate(X) if np.sum(item) != 0]
    list_idx = np.unique(list_idx)
    if len(list_idx) > 0:
        X = X[list_idx, :]
        sample_ids = [np.array(sample_ids)[list_idx]]
        sample_ids = list(sample_ids[0])
    else:
        X, sample_ids = 0, 0
    return X, sample_ids


# ---------------------------------------------------------------------------------------

def parse_files(ec_dict, input_folder, rsfolder, rspath, num_jobs):
    X, sample_ids = __preprocess(ec_dict=ec_dict, data_folder_path=input_folder,
                                 result_folder_path=os.path.join(rspath, rsfolder),
                                 num_jobs=num_jobs)
    return lil_matrix(X), sample_ids
