![Workflow](flowchart.png)

## Basic Description

This repo contains an implementation of triUMPF (**tri**ple non-negative matrix factorization with comm**U**nity
detection to **M**etabolic **P**athway in**F**erence) that combines three stages of NMF to capture relationships between
enzymes and pathways within a network followed by community detection to extract higher order structure based on the
clustering of vertices sharing similar functional features. We evaluated triUMPF performance using experimental datasets
manifesting diverse multi-label properties, including Tier 1 genomes from the BioCyc collection of organismal
Pathway/Genome Databases and low complexity microbial communities. Resulting performance metrics equaled or exceeded
other prediction methods on organismal genomes with improved prediction outcomes on multi-organism data sets.

See tutorials on the [GitHub wiki](https://github.com/hallamlab/triUMPF/wiki) page for more information and guidelines.

## Citing

If you find *triUMPF* useful in your research, please consider citing the following paper:

- M. A. Basher, Abdur Rahman, McLaughlin, Ryan J., and Hallam, Steven
  J.. **["Metabolic pathway prediction using non-negative matrix factorization with improved precision"](https://doi.org/10.1101/2020.05.27.119826)**
  , bioRxiv (2021).

## Contact

For any inquiries, please contact: [arbasher@student.ubc.ca](mailto:arbasher@student.ubc.ca)
