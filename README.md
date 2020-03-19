![Workflow](flowchart.png)

## Basic Description
This repo contains an implementation of triUMPF(**tri**ple non-negative matrix factorization with comm**u**nity detection to **m**etabolic **p**athway in**f**erence) that combines three stages of NMF to capture relationships between enzymes and pathways within a network followed by community detection to extract higher order structure based on the clustering of vertices sharing similar functional features. We evaluated triUMPF performance using experimental datasets manifesting diverse multi-label properties, including Tier 1 genomes from the BioCyc collection of organismal Pathway/Genome Databases and low complexity microbial communities. Resulting performance metrics equaled or exceeded other prediction methods on organismal genomes with improved prediction outcomes on multi-organism data sets.

## Dependencies
The codebase is tested to work under Python 3.7. To install the necessary requirements, run the following commands:

``pip install -r requirements.txt``

Basically, *triUMPF* requires following packages:
- [Anaconda](https://www.anaconda.com/)
- [NumPy](http://www.numpy.org/) (>= 1.15)
- [scikit-learn](https://scikit-learn.org/stable/) (>= 0.20)
- [pandas](http://pandas.pydata.org/) (>= 0.23)
- [NetworkX](https://networkx.github.io/) (>= 2.2)
- [scipy](https://www.scipy.org/index.html) (==1.2)


## Experimental Objects and Test Samples
Please download the following files from [Zenodo](https://zenodo.org/deposit/3711138). 
- The link contains the following preprocessed files:
    - "biocyc.pkl": an object containing the preprocessed MetaCyc database.
    - "pathway2ec.pkl": a matrix representing Pathway-EC association of size 3650 x 2526.
    - "pathway2ec_idx.pkl": the pathway2ec association indices.                        
    - "M.pkl": a sub matrix from pathway2ec.
    - "A.pkl": Pathway-Pathway interaction matrix of size 2526 x 2526.
    - "B.pkl": EC-EC interaction matrix of size 3650 x 3650.
    - "P.pkl": Pathway features matrix of size 2526 x 128.
    - "E.pkl": EC features matrix of size 3650 x 128.
    - "hin.pkl": a sample of heterogeneous information network. 
    - "pathway2vec_embeddings.npz": a sample of embeddings (nodes x dimension size). Based on your tests, you need to generate features using [pathway2vec](https://github.com/hallamlab/pathway2vec).
- We also provided pretrained models and samples for testing:
    - "golden_X.pkl": Golden dataset of size 63 x 3650. First six examples correspond to: AraCyc, EcoCyc, HumanCyc, LeishCyc, TrypanoCyc, and YeastCyc.
    - "golden_Xe.pkl": Golden dataset of size 63 x 3778. First six examples correspond to: AraCyc, EcoCyc, HumanCyc, LeishCyc, TrypanoCyc, and YeastCyc.
    - "golden_y.pkl": Golden dataset of size 63 x 2526. First six examples correspond pathways to: AraCyc, EcoCyc, HumanCyc, LeishCyc, TrypanoCyc, and YeastCyc.
    - "symbionts_X.pkl": Symbiont dataset of size 3 x 3650.
    - "symbionts_Xe.pkl": Symbiont dataset of size 3 x 3778.
    - "symbionts_y.pkl": Symbiont dataset of size 3 x 2526.
    - "cami_X.pkl": CAMI dataset of size 40 x 3650.
    - "cami_Xe.pkl": CAMI dataset of size 40 x 3778.
    - "cami_y.pkl": CAMI dataset of size 40 x 2526.
    - "hots_4_X.pkl": HOT metagenomics dataset of size 6 x 3650.
    - "hots_4_Xe.pkl": HOT metagenomics dataset of size 6 x 3778.
    - "hots_4_y.pkl": HOT metagenomics dataset of size 6 x 2526.
    - "biocyc205_tier23_9255_X.pkl": BioCyc (v20.5 tier 2 \& 3) dataset of size 9255 x 3650.
    - "biocyc205_tier23_9255_Xe.pkl": BioCyc (v20.5 tier 2 \& 3) dataset of size 9255 x 3778.
    - "biocyc205_tier23_9255_y.pkl": BioCyc (v20.5 tier 2 \& 3) dataset of size 9255 x 2526.
    - "triUMPF_X.pkl": a pretrained model using "biocyc205_tier23_9255_X.pkl" and "biocyc205_tier23_9255_y.pkl".
    - "triUMPF_X_W.pkl": a pretrained latent factors for pathways of size 2526 x 100.
    - "triUMPF_X_H.pkl": a pretrained basis matrix for ECs of size 3650 x 100.
    - "triUMPF_X_U.pkl": an auxilary matrix of size 128 x 100.
    - "triUMPF_X_V.pkl": an auxilary matrix of size 128 x 100.
    - "triUMPF_X_T.pkl": a pathway community representation matrix of size 128 x 90.
    - "triUMPF_X_C.pkl": a pathway community indicator matrix of size 2526 x 90.
    - "triUMPF_X_R.pkl": a EC community representation matrix of size 128 x 100.
    - "triUMPF_X_K.pkl": a EC community representation matrix of size 3650 x 100.
    - "triUMPF_X_L.pkl": an auxilary matrix of size 9255 x 128.
    - "triUMPF_X_Z.pkl": an auxilary matrix of size 9255 x 3650.
    - "triUMPF_e.pkl": a pretrained model using "biocyc205_tier23_9255_Xe.pkl" and "biocyc205_tier23_9255_y.pkl".
    - "triUMPF_e\*.pkl": Descriptions about the remaining matrices are same as triUMPF_X\*.

## Installation and Basic Usage
Run the following commands to clone the repository to an approriate location:

``git clone https://github.com/hallamlab/triUMPF.git``

For all experiments, navigate to ``src`` folder then run the commands of your choice. For example, to display *triUMPF*'s running options use: `python main.py --help`. It should be self-contained. 

### Preprocessing
To preprocess graphs, we provide few examples. For all examples: *--hin-file* corresponds to the desired generated file name, ending with *.pkl*.

#### Example 1
To preprocess datasets with **no noise** to the pathway2ec association matrix ("pathway2ec.pkl"), execute the following command:

``python main.py --preprocess-dataset --ssample-input-size 1 --object-name "biocyc.pkl" --pathway2ec-name "pathway2ec.pkl" --pathway2ec-pidx-name "pathway2ec_idx.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --hin-name "hin_cmt.pkl" --mdpath [Location of the features] --ospath [Location to all objects except features]``

#### Example 2
To preprocess datasets with **20% noise** to the pathway2ec association matrix ("pathway2ec.pkl"), execute the following command:

``python main.py --preprocess-dataset --ssample-input-size 0.2 --object-name "biocyc.pkl" --pathway2ec-name "pathway2ec.pkl" --pathway2ec-pidx-name "pathway2ec_idx.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --hin-name "hin_cmt.pkl" --mdpath [Location of the features] --ospath [Location to all objects except features]``

#### Example 3
To preprocess datasets with **20% noise** to the pathway2ec association (*pathway2ec.pkl*), the pathway to pathway association (*A*), and the EC to EC association (*B*) matrices, execute the following command:

``python main.py --preprocess-dataset --white-links --ssample-input-size 0.2 --object-name "biocyc.pkl" --pathway2ec-name "pathway2ec.pkl" --pathway2ec-pidx-name "pathway2ec_idx.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --hin-name "hin_cmt.pkl" --mdpath [Location of the features] --ospath [Location to all objects except features]``

### Train
For trainning, we provide few examples. 

Description about arguments in all examples: *--cutting-point* is the cutting point after which binarize operation is halted in the input data, *--M-name* is the pathway2ec association matrix file name, *--W-name* is the W parameter, *--H-name* is the H parameter, and *--model-name* corresponds the name of the model excluding any *EXTENSION*. The model name will have *.pkl* extension. The arguments *--P-name* corresponds the pathway features file name, *--E-name* is the EC features file name, *--A-name* is the pathway to pathway association file name, *--B-name* corresponds the EC to EC association file name, *--X-name* is the input space of multi-label data, and *--y-name* is the pathway space of multi-label data. For the dataset, any multi-label dataset can be employed.

**Please** do not use "triUMPF_X.pkl" or "triUMPF_e.pkl" and all the associated models related files (e.g. triUMPF_X_C.pkl, triUMPF_X_H.pkl...etc) during this step, and change the name of the pretrained models or store them in a different folder to avoid conflict.

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
For inference, we provide few examples. 

Description about arguments in all of given examples: *--cutting-point* is the cutting point after which binarize operation is halted in the input data, *--decision-threshold* corresponds the cutoff threshold for prediction, and *--model-name* corresponds the name of the model, excluding any *EXTENSION*. The model name will have *.pkl* extension.

#### Example 1
To **predict** outputs from a dataset using already trained model with decomposed *M* of 100 components while using **features** and **community**, execute the following command:

``python main.py --predict --cutting-point 3650 --decision-threshold 0.5 --X-name 'cami_Xe.pkl' --file-name "[Various results file names without extension]" --model-name "[Model name without extension]" --dspath "[Location of the dataset and to store predicted results]" --mdpath "[Location of the model]" --logpath "[Location to the log directory]" --batch 50 --num-jobs 15``

#### Example 2

To **predict** outputs and **compile pathway report** from a dataset, generated by MetaPath v2, using already trained model with decomposed *M* of 100 components while using **features** and **community**, execute the following command:

``python main.py --predict --pathway-report --cutting-point 3650 --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name 'pathway2ec.pkl' --hin-name "hin_cmt.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --X-name 'cami_Xe.pkl' --file-name "[Various results file names without extension]" --model-name "[Model name without extension]" --rsfolder "[Name of the main folder]" --dspath "[Location of the dataset and to store predicted results]" --mdpath "[Location of the model]" --rspath "[Location for storing results]" --logpath "[Location to the log directory]" --batch 50 --num-jobs 15``

where *--pathway-report* enables to generate a detailed report for pathways for each instance.

## Citing
If you find *triUMPF* useful in your research, please consider citing the following paper:
- M. A. Basher, Abdur Rahman, McLaughlin, Ryan J., and Hallam, Steven J.. **["Metabolic pathway inference using non-negative matrix factorization with community detection."](https://github.com/arbasher/xXx.pdf)**, bioRxiv (2020).

## Contact
For any inquiries, please contact: [arbasher@alumni.ubc.ca](mailto:arbasher@alumni.ubc.ca)

## Upcoming features
- Learning based on subsampling approaches.
