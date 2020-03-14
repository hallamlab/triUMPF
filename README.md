![Workflow](flowchart.png)

## Basic Description
This repo contains an implementation of triUMPF(**tri**ple non-negative matrix factorization with comm**u**nity detection to **m**etabolic **p**athway in**f**erence) that combines three stages of NMF to capture relationships between enzymes and pathways within a network followed by community detection to extract higher order structure based on the clustering of vertices sharing similar functional features. We evaluated triUMPF performance using experimental datasets manifesting diverse multi-label properties, including Tier 1 genomes from the BioCyc collection of organismal Pathway/Genome Databases and low complexity microbial communities. Resulting performance metrics equaled or exceeded other prediction methods on organismal genomes with improved prediction outcomes on multi-organism data sets.

## Dependencies

- *triUMPF* is tested to work under Python 3.5
- [Anaconda](https://www.anaconda.com/)
- [NumPy](http://www.numpy.org/) (>= 1.15)
- [scikit-learn](https://scikit-learn.org/stable/) (>= 0.20)
- [pandas](http://pandas.pydata.org/) (>= 0.23)
- [NetworkX](https://networkx.github.io/) (>= 2.2)
- [scipy](https://www.scipy.org/index.html)

## Objects
- Please download the files: "biocyc.pkl", "pathway2ec.pkl", "pathway2ec_idx.pkl", "biocyc_X.pkl", "biocyc_Xe.pkl", "biocyc_y.pkl", "cami_X.pkl", "cami_Xe.pkl", "cami_y.pkl", "M.pkl", "P.pkl", "E.pkl", "A.pkl", and "B.pkl" from: [HallamLab](https://github.com/hallamlab)
- You need to generate a heterogeneous information network. A preprocessed hin file can be obtained for the experimental purposes from [hin_cmt.pkl](https://github.com/hallamlab).
- Also, please do provide features obtained from [pathway2vec](https://github.com/hallamlab/pathway2vec). A sample of features can be obtained for the experimental purposes from [path2vec_cmt_tf_embeddings.npz](https://github.com/hallamlab).

## Basic Usage
To display *triUMPF*'s running options, use: `python main.py --help`. It should be self-contained. 
### Preprocessing graph

#### Example 1
To preprocess datasets with **no noise** to the pathway2ec association matrix ("pathway2ec.pkl"), execute the following command:

``python main.py --preprocess-dataset --ssample-input-size 1 --object-name "biocyc.pkl" --pathway2ec-name "pathway2ec.pkl" --pathway2ec-pidx-name "pathway2ec_idx.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --hin-name "hin_cmt.pkl" --mdpath [Location of the features] --ospath [Location to all objects except features]``

where *--hin-file* corresponds to the hin file name, ending with *.pkl*.

#### Example 2
To preprocess datasets with **20% noise** to the pathway2ec association matrix ("pathway2ec.pkl"), execute the following command:

``python main.py --preprocess-dataset --ssample-input-size 0.2 --object-name "biocyc.pkl" --pathway2ec-name "pathway2ec.pkl" --pathway2ec-pidx-name "pathway2ec_idx.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --hin-name "hin_cmt.pkl" --mdpath [Location of the features] --ospath [Location to all objects except features]``

where *--hin-file* corresponds to the hin file name, ending with *.pkl*.

#### Example 3
To preprocess datasets with **20% noise** to the pathway2ec association (*pathway2ec.pkl*), the pathway to pathway association (*A*), and the EC to EC association (*B*) matrices, execute the following command:

``python main.py --preprocess-dataset --white-links --ssample-input-size 0.2 --object-name "biocyc.pkl" --pathway2ec-name "pathway2ec.pkl" --pathway2ec-pidx-name "pathway2ec_idx.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --hin-name "hin_cmt.pkl" --mdpath [Location of the features] --ospath [Location to all objects except features]``

where *--hin-file* corresponds to the hin file name, ending with *.pkl*.


### Train
For trainning, we provide few examples. The description of arguments: *--cutting-point* is the cutting point after which binarize operation is halted in the input data, *--M-name* is the pathway2ec association matrix file name, *--W-name* is the W parameter, *--H-name* is the H parameter, and *--model-name* corresponds the name of the model excluding any *EXTENSION*. The model name will have *.pkl* extension. The arguments *--P-name* corresponds the pathway features file name, *--E-name* is the EC features file name, *--A-name* is the pathway to pathway association file name, *--B-name* corresponds the EC to EC association file name, *--X-name* is the input space of multi-label data, and *--y-name* is the pathway space of multi-label data. For the dataset, any multi-label dataset can be employed.

#### Example 1
To **decompose** *M* of 100 components, execute the following command:

``python main.py --train --num-components 100 --lambdas 0.01 0.01 0.01 0.01 0.001 10 --M-name "M.pkl" --model-name "[Model name without extension]" --mdpath "[Location of the model]" --logpath "[Location to the log directory]" --batch 50 --max-inner-iter 5 --num-epochs 100 --num-jobs 2``

#### Example 2
To **decompose** *M* of 100 components by using **features**, execute the following command:

``python main.py --train --fit-features --num-components 100 --lambdas 0.01 0.01 0.01 0.01 0.001 10 --M-name "M.pkl" --P-name "P.pkl" --E-name "E.pkl" --model-name "[Model name without extension]" --mdpath "[Location of the model]" --logpath "[Location to the log directory]" --batch 50 --max-inner-iter 5 --num-epochs 100 --num-jobs 2``

#### Example 3
If you wish to train multi-label dataset by **decomposing** *M* of 100 components while using **features** and **community**, execute the following command:

``python main.py --train --fit-features --fit-comm --binarize --use-external-features --cutting-point 3650 --num-components 100 --lambdas 0.01 0.01 0.01 0.01 0.001 10 --M-name "M.pkl" --P-name "P.pkl" --E-name "E.pkl"  --A-name "A.pkl" --B-name "B.pkl" --X-name "biocyc_Xe.pkl" --y-name "biocyc_y.pkl" --model-name "[Model name without extension]" --mdpath "[Location of the model]" --dspath "[Location of the dataset]" --logpath "[Location to the log directory]" --batch 50 --max-inner-iter 5 --num-epochs 100 --num-jobs 2``

#### Example 4
If you wish to use the previously **decomposed** *M* of 100 components to train multi-label dataset while using **features** and **community**, execute the following command:

``python main.py --train --no-decomposition --fit-features --fit-comm --binarize --use-external-features --cutting-point 3650 --num-components 100 --lambdas 0.01 0.01 0.01 0.01 0.001 10 --W-name "[Generated .pkl W file]" --H-name "[Generated .pkl H file]"  --P-name "P.pkl" --E-name "E.pkl"  --A-name "A.pkl" --B-name "B.pkl" --X-name "biocyc_Xe.pkl" --y-name "biocyc_y.pkl" --model-name "[Model name without extension]" --mdpath "[Location of the model]" --dspath "[Location of the dataset]" --logpath "[Location to the log directory]" --batch 50 --max-inner-iter 5 --num-epochs 100 --num-jobs 2``

### Predict
For inference, we provide few examples:
#### Example 1
To **predict** outputs from a dataset using already trained model with decomposed *M* of 100 components while using **features** and **community**, execute the following command:

``python main.py --predict --cutting-point 3650 --decision-threshold 0.5 --X-name 'cami_Xe.pkl' --file-name "[Various results file names without extension]" --model-name "[Model name without extension]" --dspath "[Location of the dataset and to store predicted results]" --mdpath "[Location of the model]" --logpath "[Location to the log directory]" --batch 50 --num-jobs 15``

where *--cutting-point* is the cutting point after which binarize operation is halted in the input data, *--decision-threshold* corresponds the cutoff threshold for prediction, and *--model-name* corresponds the name of the model, excluding any *EXTENSION*. The model name will have *.pkl* extension. For the dataset, any multi-label dataset can be employed.

#### Example 2

To **predict** outputs and **compile pathway report** from a dataset, generated by MetaPath v2, using already trained model with decomposed *M* of 100 components while using **features** and **community**, execute the following command:

``python main.py --predict --pathway-report --cutting-point 3650 --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name 'pathway2ec.pkl' --hin-name "hin_cmt.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --X-name 'cami_Xe.pkl' --file-name "[Various results file names without extension]" --model-name "[Model name without extension]" --rsfolder "[Name of the main folder]" --dspath "[Location of the dataset and to store predicted results]" --mdpath "[Location of the model]" --rspath "[Location for storing results]" --logpath "[Location to the log directory]" --batch 50 --num-jobs 15``

where *--pathway-report* enables to generate a detailed report for pathways for each instance, *--cutting-point* is the cutting point after which binarize operation is halted in the input data, *--decision-threshold* corresponds the cutoff threshold for prediction, and *--model-name* corresponds the name of the model, excluding any *EXTENSION*. The model name will have *.pkl* extension. For the dataset, any multi-label dataset can be employed.


## Citing
If you employ *triUMPF* in your research, please consider citing the following paper presented at bioRxiv 2020:
- M. A. Basher, Abdur Rahman, McLaughlin, Ryan J., and Hallam, Steven J.. **["Metabolic pathway inference using non-negative matrix factorization with community detection."](https://github.com/atante/xXx.pdf)**, Proceedings of bioRxiv (2020).

## Contact
For any inquiries, please contact: [arbasher@alumni.ubc.ca](mailto:arbasher@alumni.ubc.ca)
