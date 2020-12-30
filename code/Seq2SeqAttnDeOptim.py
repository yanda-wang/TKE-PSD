import sys
import skorch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch import optim
from Optimization import MedRecSeq2SeqTrainer, MedRecSeq2SeqAttnDe
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import load as optim_load
from skopt.callbacks import CheckpointSaver

from Parameters import Params

params = Params()
PATIENT_RECORDS_ORDERED_FILE = params.PATIENT_RECORDS_ORDERED_FILE
CONCEPTID_FILE = params.CONCEPTID_FILE  # 'data/voc_final.pkl'
EHR_MATRIX_FILE = params.EHR_MATRIX_FILE  # 'data/ehr_adj_final.pkl'
DEVICE = params.device  # torch.device("cuda" if USE_CUDA else "cpu")
MEDICATION_COUNT = params.MEDICATION_COUNT  # 153
DIAGNOSES_COUNT = params.DIAGNOSES_COUNT  # 1960
PROCEDURES_COUNT = params.PROCEDURES_COUNT  # 1432

OPT_SPLIT_TAG_ADMISSION = params.OPT_SPLIT_TAG_ADMISSION  # -1
OPT_SPLIT_TAG_VARIABLE = params.OPT_SPLIT_TAG_VARIABLE  # -2
OPT_MODEL_MAX_EPOCH = params.OPT_MODEL_MAX_EPOCH

SEQ_MAX_LENGTH = params.SEQUENCE_MAX_LENGTH
TEACHING_FORCE_RATE = params.TEACHING_FORCE_RATE
ENCODER_HIDDEN_MAX_LENGTH = params.ENCODER_HIDDEN_MAX_LENGTH

TRAIN_RATIO = params.train_ratio
TEST_RATIO = params.test_ratio

LOG_FILE = 'data/log/seq2seq_single_attn_seploss_de.log'
OPTIMIZE_RESULT_FILE = 'data/hyper-tuning/checkpoint_AttnDe.pkl'


def concatenate_single_admission(records):
    records = records[:4]  # remove potentials, save diagnose, procedures, medications, medications (ordered)
    x = records[0]
    for item in records[1:]:
        x = x + [OPT_SPLIT_TAG_VARIABLE] + item
    return x


def concatenate_all_admissions(records):
    x = concatenate_single_admission(records[0])
    for admission in records[1:]:
        current_adm = concatenate_single_admission(admission)
        x = x + [OPT_SPLIT_TAG_ADMISSION] + current_adm
    return x


def get_x_y(patient_records):
    x, y = [], []
    for patient in patient_records:
        for idx, adm in enumerate(patient):
            current_records = patient[:idx + 1]
            current_x = concatenate_all_admissions(current_records)
            x.append(np.array(current_x))
            target = adm[3] + [params.EOS_id]  # ordered medication sequence
            y.append(np.array(target))
    return np.array(x), np.array(y)


def get_data(patient_records_file):
    patient_records = pd.read_pickle(patient_records_file)
    split_point = int(len(patient_records) * TRAIN_RATIO)
    test_count = int(len(patient_records) * TEST_RATIO)
    train = patient_records[:split_point]
    test = patient_records[split_point:split_point + test_count]

    train_x, train_y = get_x_y(train)
    test_x, test_y = get_x_y(test)
    return train_x, train_y, test_x, test_y


def get_metric(y_predict, y_target):
    count = 0
    avg_precision = 0.0
    avg_recall = 0.0
    for yp, yt in zip(y_predict, y_target):
        if yp.shape[0] == 0:
            precision = 0
            recall = 0
        else:
            intersection = list(set(yp.tolist()) & set(yt.tolist()))
            precision = float(len(intersection)) / len(set(yp.tolist()))
            recall = float(len(intersection)) / len(set(yt.tolist()))
        count += 1
        avg_precision += precision
        avg_recall += recall
    avg_precision = avg_precision / count
    avg_recall = avg_recall / count
    if avg_precision + avg_recall == 0:
        return 0
    return 2.0 * avg_precision * avg_recall / (avg_precision + avg_recall)


search_space = [Categorical(categories=['64', '128', '200', '256', '300', '400'], name='dimension'),
                Integer(low=1, high=5, name='encoder_n_layers'),
                Real(low=0, high=1, name='encoder_embedding_dropout_rate'),
                Real(low=0, high=1, name='encoder_gru_dropout_rate'),

                Real(low=0, high=1, name='decoder_dropout_rate_input_med'),

                Real(low=1e-6, high=1e-3, prior='log-uniform', name='optimizer_encoder_learning_rate'),
                Real(low=1e-6, high=1e-3, prior='log-uniform', name='optimizer_decoder_learning_rate')
                ]


@use_named_args(dimensions=search_space)
def fitness(dimension, encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate,
            decoder_dropout_rate_input_med, optimizer_encoder_learning_rate,
            optimizer_decoder_learning_rate):
    input_size = int(dimension)
    hidden_size = int(dimension)

    model = MedRecSeq2SeqTrainer(criterion=nn.NLLLoss, optimizer_encoder=optim.Adam, optimizer_decoder=optim.Adam,
                                 max_epochs=OPT_MODEL_MAX_EPOCH, batch_size=1, train_split=None,
                                 callbacks=[skorch.callbacks.ProgressBar(batches_per_epoch='auto'), ], device=DEVICE,
                                 module=MedRecSeq2SeqAttnDe, module__device=DEVICE, module__input_size=input_size,
                                 module__hidden_size=hidden_size, module__output_size=MEDICATION_COUNT,
                                 module__max_length=SEQ_MAX_LENGTH, module__teaching_force_rate=TEACHING_FORCE_RATE,
                                 module__encoder_hidden_max_length=ENCODER_HIDDEN_MAX_LENGTH,

                                 module__encoder__diagnoses_count=DIAGNOSES_COUNT,
                                 module__encoder__procedures_count=PROCEDURES_COUNT,
                                 module__encoder__n_layers=encoder_n_layers.item(),
                                 module__encoder__embedding_dropout_rate=encoder_embedding_dropout_rate,
                                 module__encoder__gru_dropout_rate=encoder_gru_dropout_rate,

                                 module__decoder__dropout_rate_input_med=decoder_dropout_rate_input_med,

                                 optimizer_encoder_learning_rate=optimizer_encoder_learning_rate,
                                 optimizer_decoder_learning_rate=optimizer_decoder_learning_rate,
                                 )

    train_x, train_y, test_x, test_y = get_data(PATIENT_RECORDS_ORDERED_FILE)

    model.fit(train_x, train_y)
    predict_y = model.predict(test_x)
    metric = get_metric(predict_y, test_y)

    print('*' * 30)
    print('hyper-parameters')
    print('input size:', input_size)
    print('encoder_n_layers:', encoder_n_layers)
    print('encoder_embedding_dropout_rate:', encoder_gru_dropout_rate)
    print('encoder_gru_dropout_rate:', encoder_gru_dropout_rate)
    print('encoder_optimizer_lr:{0:.1e}'.format(optimizer_encoder_learning_rate))

    print('decoder_dropout_rate_input_med:', decoder_dropout_rate_input_med)
    print('decoder_optimizer_lr:{0:.1e}'.format(optimizer_decoder_learning_rate))

    print('f1:{0:.4f}'.format(metric))
    print()

    return -metric


def optimize(n_calls):
    sys.stdout = open(LOG_FILE, 'a')
    checkpoint_saver = CheckpointSaver(OPTIMIZE_RESULT_FILE, compress=9)

    # ######load existing optimization result and carry on
    # optim_result = optim_load(OPTIMIZE_RESULT_FILE)
    # examined_values = optim_result.x_iters
    # observed_values = optim_result.func_vals
    # result = gp_minimize(fitness, search_space, n_calls=n_calls, verbose=True, callback=[checkpoint_saver],
    #                      x0=examined_values, y0=observed_values, n_initial_points=-len(examined_values))

    result = gp_minimize(fitness, search_space, n_calls=n_calls, verbose=True, callback=[checkpoint_saver])

    print('**********************************')
    print('best result:')
    print('f1:', -result.fun)
    print('optimal hyper-parameters')

    space_dim_name = [item.name for item in search_space]
    for hyper, value in zip(space_dim_name, result.x):
        print(hyper, value)

    sys.stdout.close()


if __name__ == '__main__':
    optimize(11)
