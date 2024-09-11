# TOKEPER

# Overview

This is the implement for paper Topological Knowledge Enhanced Personalized Ranking Model for Sequential Medication Recommendation which is a model that combines global topological knowledge of medications and personalized patient conditions to determine the order of medications for sequential recommendation.

TOKEPER firstly conduct medication recommendation to generate drugs as sets (seq2set), and uses the probabilities of these drugs as the predict potential, then TOKEPER calculates the topological potential of medications based on their co-occurrence. These two kinds of potential are combined by weighted sum, and the final results are used to determine the order of medications for sequential recommendation (seq2seq). In addition, TOKEPER propsoes Occurrence-Based (OB) Beam search, a modified Beam search that uses the occurrence frequency of medications in the predicted sequences to change the probabilitis of drugs, so that redundant and duplicate drugs could be avoided.

# Requirement

Pytorch 1.1

Python 3.7

# Data

Experiments are conducted based on [MIMIC-III](https://mimic.physionet.org), a real-world Electronic Healthcare Records (EHRs) dataset. MIMIC-III collects clinical information related to over 40,000 patients, and diagnoses and procedures are used as inputs of SARMR, while the medications prescribed in the first 24 hours of each admission are selected out as ground truths.

Patient records are selected from the raw data, and each patient is represented by the clinical informaiton of each admission, which include diagnoses, procedures, and medications. These admissions are ordered chronologically. An example of patient records is presented as follows, which contains two admissions.

\[\[\['4239', '5119', '78551', '4589'\],\['3731', '8872', '3893'\],\['N02B', 'A01A', 'A02B', 'A06A', 'B05C', 'A12'\]\],
  \[\['7455', '45829', 'V1259', '2724'\],\['3571', '3961', '8872'\],\['N02B', 'A01A', 'A02B', 'A06A', 'A12A'\]\]\]

Items in patient records are then assigned a identical code, so that the raw records are transformed into the following format, which would act as the input of TOKEPER.

\[\[\[2, 3, 4, 5, 6, 7, 8, 9\], \[2, 3, 4\], \[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17\]\],
  \[\[10, 11, 12, 9\], \[5, 6, 3\], \[2, 3, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20\]\]\]


# Code

Auxiliary.py: data preprocessing.

BeamSearchOptimization.py: hyper-parameters tuning based on Gaussian Process for OB Beam.

Evaluation.py: model evaluation.

Networks.py: modules for the neural network.

Optimization.py: basic modules for hyper-parameters tuning.

Parameters.py: global parameters for the model.

Seq2SeqAttnDeOptim.py: hyper-parameters tuning based on Gaussian Process for the sequential recommendation.

Seq2SetOptim.py: hyper-parameters tuning based on Gaussian Process for the generation of medication sets.

SequenceOrder.py: generation of different medication sequences.

Training.py: model training.

To conduct the sequential medication recommendation, one can follows the following steps:
1. Conducting data preprocessing to get the input, the code could be found in Auxiliary.py.
2. Obtaining the optimal hyper-parameters for the seq2set model by using the code in Seq2SetOptim.py.
3. Training the seq2set model to obtain predict potential for medications, and the code could be found in Training.py.
4. Using the code in SequenceOrder.py to generating the ground truth medication sequences for the sequential recommendation.
5. Obtaining the optimal hyper-parameters for the seq2seq model by using the code in Seq2SeqAttnDeOptim.py.
6. Using Training.py to train the seq2seq model.
7. Determing the hyper-parameters for OB Beam by using BeamSearchOptimization.py.
