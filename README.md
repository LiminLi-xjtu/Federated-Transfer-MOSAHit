# Federated-Transfer-MOSAHit
Federated transfer learning with differential privacy for multi-omics survival analysis

MOSAHit method, trained using federated transfer learning, enables the learning of a more robust multi-omics survival prediction model for a local target cancer with limited training data by effectively leveraging multi-omics data of related cancers distributed across multiple institutions, while preserving individual privacy.

# Environment Installation

$ conda env create -f environment.yml

$ source activate survival_analysis

# Data Download

The TCGA datasets are publicly available and can be obtained from the UCSC Xena database (https://xena.ucsc.edu/). The original data can be accessed at: https://www.cancer.gov/ccg/research/genome-sequencing/tcga.


# Data Preprocessing

#### cd Federated-Transfer-MOSAHit/Datasets

The python program (extract_RNASeq_expression.py) summarized the gene expression of patients with the same type of cancer into a csv file. 

#### python extract_RNASeq_expression.py

#### python extract_miRNA_expression.py

The python program (RNASeq_variable_filter.py) imputed missing data, filtered out noise-sensitive features (those whose values ​​remained almost constant across samples), and performed a logarithmic transformation.

#### python RNASeq_variable_filter.py

#### python microRNA_variable_filter.py

The python program (clinical_preprocess.py)  aggregated the survival data of cancer patients (such as censoring indicator δ and observed time O).

#### python clinical_preprocess.py

The python program (matched_patients.py) only kept the patients with matched gene expression, microRNA expression data and survival data.

#### python matched_patients.py

# Run the main routine

#### git clone https://github.com/LiminLi-xjtu/Federated-Transfer-MOSAHit.git

#### cd Federated-Transfer-MOSAHit

#### python DPFL.py

