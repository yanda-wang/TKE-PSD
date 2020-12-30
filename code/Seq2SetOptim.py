import skorch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch import optim
from Optimization import MedRecSeq2Set, MedRecTrainer
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

from Parameters import Params

params = Params()
PATIENT_RECORDS_FILE = params.PATIENT_RECORDS_FILE  # 'data/records_final.pkl'
CONCEPTID_FILE = params.CONCEPTID_FILE  # 'data/voc_final.pkl'
EHR_MATRIX_FILE = params.EHR_MATRIX_FILE  # 'data/ehr_adj_final.pkl'
DEVICE = params.device  # torch.device("cuda" if USE_CUDA else "cpu")
MEDICATION_COUNT = params.MEDICATION_COUNT  # 153
DIAGNOSES_COUNT = params.DIAGNOSES_COUNT  # 1960
PROCEDURES_COUNT = params.PROCEDURES_COUNT  # 1432

OPT_SPLIT_TAG_ADMISSION = params.OPT_SPLIT_TAG_ADMISSION  # -1
OPT_SPLIT_TAG_VARIABLE = params.OPT_SPLIT_TAG_VARIABLE  # -2
OPT_MODEL_MAX_EPOCH = params.OPT_MODEL_MAX_EPOCH

TRAIN_RATIO = params.train_ratio
TEST_RATIO = params.test_ratio

LOG_FILE = 'data/log/seq2set_single_attn_seploss_v2.log'


def concatenate_single_admission(records):
    return records[0] + [OPT_SPLIT_TAG_VARIABLE] + records[1] + [OPT_SPLIT_TAG_VARIABLE] + records[2]


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
            target = adm[2]
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

                Integer(low=1, high=20, name='decoder_hop'),
                Real(low=0, high=1, name='decoder_dropout_rate'),
                Categorical(categories=['dot', 'general', 'concat'], name='decoder_attn_type_kv'),
                Categorical(categories=['dot', 'general', 'concat'], name='decoder_attn_type_embedding'),
                Real(low=1e-6, high=1e-3, prior='log-uniform', name='optimizer_encoder_learning_rate'),
                Real(low=1e-6, high=1e-3, prior='log-uniform', name='optimizer_decoder_learning_rate')
                ]


@use_named_args(dimensions=search_space)
def fitness(dimension, encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate, decoder_hop,
            decoder_dropout_rate, decoder_attn_type_kv, decoder_attn_type_embedding, optimizer_encoder_learning_rate,
            optimizer_decoder_learning_rate):
    ehr_matrix = np.load(EHR_MATRIX_FILE)

    input_size = int(dimension)
    hidden_size = int(dimension)

    model = MedRecTrainer(criterion=nn.BCEWithLogitsLoss, optimizer_encoder=optim.Adam,
                          optimizer_decoder=optim.Adam, max_epochs=OPT_MODEL_MAX_EPOCH, batch_size=1,
                          train_split=None, callbacks=[skorch.callbacks.ProgressBar(batches_per_epoch='auto'), ],
                          device=DEVICE, module=MedRecSeq2Set, module__device=DEVICE,
                          module__input_size=input_size, module__hidden_size=hidden_size,
                          module__encoder__diagnoses_count=DIAGNOSES_COUNT,
                          module__encoder__procedures_count=PROCEDURES_COUNT,
                          module__encoder__n_layers=encoder_n_layers.item(),
                          module__encoder__embedding_dropout_rate=encoder_embedding_dropout_rate,
                          module__encoder__gru_dropout_rate=encoder_gru_dropout_rate,
                          module__encoder__bidirectional=False,

                          module__decoder__output_size=MEDICATION_COUNT,
                          module__decoder__medication_count=MEDICATION_COUNT, module__decoder__hop=decoder_hop,

                          module__decoder__dropout_rate=decoder_dropout_rate,
                          module__decoder__attn_type_kv=decoder_attn_type_kv,
                          module__decoder__attn_type_embedding=decoder_attn_type_embedding,
                          module__decoder__ehr_adj=ehr_matrix,
                          optimizer_encoder__lr=optimizer_encoder_learning_rate,
                          optimizer_encoder__weight_decay=0,
                          optimizer_decoder__lr=optimizer_decoder_learning_rate,
                          optimizer_decoder__weight_decay=0)

    train_x, train_y, test_x, test_y = get_data(PATIENT_RECORDS_FILE)
    model.fit(train_x, train_y)
    predict_y = model.predict(test_x)  # np.array,dim=(#patients,#medications for each patient)
    metric = get_metric(predict_y, test_y)

    print('**********************************')
    print('hyper-parameters')
    print('input size:', input_size)
    print('encoder_n_layers:', encoder_n_layers)
    print('encoder_embedding_dropout_rate:', encoder_embedding_dropout_rate)
    print('encoder_gru_dropout_rate:', encoder_gru_dropout_rate)
    # print('encoder_bidirectional:', encoder_bidirectional)
    print('encoder_optimizer_lr:{0:.1e}'.format(optimizer_encoder_learning_rate))
    # print('encoder_optimizer_regular_lambda:', optimizer_encoder_regular_lambda)

    print('decoder_hop:', decoder_hop)
    print('decoder_dropout_rate:', decoder_dropout_rate)
    print('decoder_attn_type_kv:', decoder_attn_type_kv)
    print('decoder_attn_type_embedding:', decoder_attn_type_embedding)
    print('decoder_optimizer_lr:{0:.1e}'.format(optimizer_decoder_learning_rate))
    # print('decoder_optimizer_regular_lambda:', optimizer_decoder_regular_lambda)
    print()
    print('f1: {0:.4f}'.format(metric))

    log_file = open(LOG_FILE, 'a+')
    log_file.write('input_size:' + str(input_size) + '\n')
    log_file.write('encoder_n_layers:' + str(encoder_n_layers) + '\n')
    log_file.write('encoder_embedding_dropout_rate:' + str(encoder_embedding_dropout_rate) + '\n')
    log_file.write('encoder_gru_dropout_rate:' + str(encoder_gru_dropout_rate) + '\n')
    log_file.write('encoder_optimizer_learning_rate:' + str(optimizer_encoder_learning_rate) + '\n')
    # log_file.write('encoder_optimizer_weight_decay:' + str(optimizer_encoder_regular_lambda) + '\n')

    log_file.write('decoder_hop:' + str(decoder_hop) + '\n')
    log_file.write('decoder_dropout_rate:' + str(decoder_dropout_rate) + '\n')
    log_file.write('decoder_attn_type_kv:' + decoder_attn_type_kv + '\n')
    log_file.write('decoder_attn_type_embedding:' + decoder_attn_type_embedding + '\n')
    log_file.write('decoder_optimizer_learning_rate:' + str(optimizer_decoder_learning_rate) + '\n')
    # log_file.write('decoder_optimizer_weight_decay:' + str(optimizer_decoder_regular_lambda) + '\n')
    log_file.write('f1: {0:.4f}'.format(metric) + '\n')
    log_file.write('*************************************\n')
    log_file.close()

    return -metric


def optimize(n_calls):
    result = gp_minimize(fitness, search_space, n_calls=n_calls, verbose=True)
    print('**********************************')
    print('best result:')
    print('f1:', -result.fun)
    print('optimal hyper-parameters')

    log_file = open(LOG_FILE, 'a+')
    log_file.write('best result:\n')
    log_file.write(str(-result.fun))
    log_file.write('\n')
    log_file.write('optimal hyper-parameters\n')
    space_dim_name = [item.name for item in search_space]
    for hyper, value in zip(space_dim_name, result.x):
        print(hyper, value)
        log_file.write(hyper)
        log_file.write(':')
        log_file.write(str(value))
        log_file.write('\n')

    log_file.close()


if __name__ == "__main__":
    optimize(11)
