import torch
import numpy as np
import pandas as pd

from Networks import EncoderSeq, DecoderSeqAttnDe
from Evaluation import BeamOccur
from Evaluation import EvaluationUtil

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from tqdm import tqdm
from torch.autograd import Variable

from Parameters import Params

params = Params()
PATIENT_RECORDS_ORDERED_FILE = params.PATIENT_RECORDS_ORDERED_FILE
CONCEPTID_FILE = params.CONCEPTID_FILE  # 'data/voc_final.pkl'
EHR_MATRIX_FILE = params.EHR_MATRIX_FILE  # 'data/ehr_adj_final.pkl'
DEVICE = params.device  # torch.device("cuda" if USE_CUDA else "cpu")
MEDICATION_COUNT = params.MEDICATION_COUNT  # 153
DIAGNOSES_COUNT = params.DIAGNOSES_COUNT  # 1960
PROCEDURES_COUNT = params.PROCEDURES_COUNT  # 1432
SEQUENCE_MAX_LENGTH = params.SEQUENCE_MAX_LENGTH
ENCODER_HIDDEN_MAX_LENGTH = params.ENCODER_HIDDEN_MAX_LENGTH
PREDICT_SEQUENCE_MAX_LENGTH = params.SEQUENCE_MAX_LENGTH

MODEL_FILE = 'data/model/seq2seq_maxlength40_teach_init0.8_change_40_constant_60/2_200_200_0.9_0.1_AttnDe/0.33924615_0.25448997_5.89e-06_0.24392542_1.96e-05/seq2seq_110_1355530_2.3730_0.6312.checkpoint'


def load_models(model_file, input_size=200, hidden_size=200, encoder_n_layers=2,
                encoder_input_embedding_dropout_rate=0.33924615, encoder_gru_dropout_rate=0.25448997,
                decoder_input_embedding_dropout_rate=0.24392542):
    checkpoint = torch.load(model_file)
    encoder = EncoderSeq(DEVICE, input_size, hidden_size, DIAGNOSES_COUNT, PROCEDURES_COUNT,
                         encoder_n_layers, encoder_input_embedding_dropout_rate, encoder_gru_dropout_rate, False)
    decoder = DecoderSeqAttnDe(hidden_size, MEDICATION_COUNT, decoder_input_embedding_dropout_rate,
                               ENCODER_HIDDEN_MAX_LENGTH, DEVICE)

    encoder_sd = checkpoint['encoder']
    decoder_sd = checkpoint['decoder']
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)
    encoder.eval()
    decoder.eval()

    return encoder, decoder


def load_data(patient_records_file):
    patient_records = pd.read_pickle(patient_records_file)
    split_point = int(len(patient_records) * params.train_ratio)
    test_count = int(len(patient_records) * params.test_ratio)
    patient_records_train = patient_records[:split_point]
    patient_records_test = patient_records[split_point:split_point + test_count]
    patient_records_validation = patient_records[split_point + test_count:split_point + test_count + test_count]

    return patient_records_train, patient_records_test, patient_records_validation


def evaluateIters(encoder, decoder, patient_records, beam_size, beam_minus_weight):
    total_metric_precision = 0.0
    total_metric_recall = 0.0
    total_metric_f1 = 0.0
    count = 0
    predict_distinct_rate = []

    evaluate_utils = EvaluationUtil()

    for i, patient in enumerate(tqdm(patient_records)):
        for idx, adm in enumerate(patient):
            count += 1
            current_records = patient[:idx + 1]
            target_medication = adm[3]
            # dim(query)=(1,hidden_size),dim(encoder_output)=(encoder_output_max_length,hidden_size)
            query, encoder_output_tmp = encoder(current_records)
            encoder_output = torch.zeros(ENCODER_HIDDEN_MAX_LENGTH, encoder.hidden_size, device=DEVICE)
            if encoder_output_tmp.size(0) <= ENCODER_HIDDEN_MAX_LENGTH:
                encoder_output[:encoder_output_tmp.size(0)] = encoder_output_tmp
            else:
                encoder_output = encoder_output_tmp[-ENCODER_HIDDEN_MAX_LENGTH:]

            beam = BeamOccur(beam_size, beam_minus_weight, params.SOS_id, params.EOS_id, MEDICATION_COUNT, DEVICE)
            decoder_input = beam.get_current_state().view(1, -1)  # dim=(1,1)
            decoder_hidden = Variable(query.unsqueeze(1).data)  # dim=(1,1,hidden_size)

            # dim(decoder_output)=(1,medication_count), dim(decoder_hidden)=(1,1,hidden_size)
            decoder_output, decoder_hidden, output_probs = decoder(decoder_input, decoder_hidden, encoder_output)
            output_probs = torch.from_numpy(output_probs)

            if not beam.advance(decoder_output, output_probs):
                decoder_hidden = Variable(decoder_hidden.data.repeat(1, beam_size, 1))  # dim=(1,beam_size,hidden)
                for _ in range(PREDICT_SEQUENCE_MAX_LENGTH - 1):
                    decoder_input = beam.get_current_state()  # dim=(beam_size)
                    decoder_hidden.data.copy_(decoder_hidden.data.index_select(1, beam.get_current_origin()))
                    # for each translation
                    new_output = []
                    new_hidden = []
                    new_probs = []
                    for input, hidden in zip(decoder_input, decoder_hidden[0]):
                        current_output, current_hidden, current_probs = decoder(input.view(1, -1),
                                                                                hidden.view(1, 1, -1),
                                                                                encoder_output)
                        new_output.append(current_output)  # append dim=(1,medication_count)
                        new_hidden.append(current_hidden[0][0])  # append dim=(hidden_size)
                        new_probs.append(current_probs[0])  # append dim=(medication_count)
                    # merge information
                    log_probs = torch.stack(new_output).squeeze(1)  # dim=(beam_size,medication_count)
                    decoder_hidden = torch.stack(new_hidden).unsqueeze(0)  # dim=(1,beam_size,hidden_size)
                    probs = np.stack(new_probs)
                    probs = torch.from_numpy(probs).to(DEVICE)  # dim=(beam_size,medication_count)
                    if beam.advance(log_probs, probs):
                        break

            n_best = 1
            scores, ks = beam.sort_best()
            scores = scores[:n_best]
            hyps = [beam.get_hyp(k) for k in ks[:n_best]]
            hyp = hyps[0]
            predict_token = torch.stack(hyp).detach().cpu().numpy()

            predict_token = [item for item in predict_token if item != params.EOS_id]
            precision = evaluate_utils.metric_precision(predict_token, target_medication)
            recall = evaluate_utils.metric_recall(predict_token, target_medication)
            f1 = evaluate_utils.metric_f1(precision, recall)

            total_metric_precision += precision
            total_metric_recall += recall
            total_metric_f1 += f1

            predict_distinct_rate.append(float(len(set(predict_token)) / len(predict_token)))

    precision_avg = total_metric_precision / count
    recall_avg = total_metric_recall / count
    f1_avg = total_metric_f1 / count

    return f1_avg


search_space = [Integer(low=1, high=30, name='beam_size'),
                Real(low=0, high=10, name='prob_minus_weight')
                ]


@use_named_args(search_space)
def fitness(beam_size, prob_minus_weight):
    encoder, decoder = load_models(MODEL_FILE)
    patient_records_train, patient_records_test, patient_records_validation = load_data(PATIENT_RECORDS_ORDERED_FILE)
    metric = evaluateIters(encoder, decoder, patient_records_validation, beam_size.item(), prob_minus_weight)

    print('beam_size:', beam_size)
    print('prob_minus_weight:', prob_minus_weight)
    print('f1:{0:.4f}'.format(metric))
    return -metric


def optimize(n_calls):
    result = gp_minimize(fitness, search_space, n_calls=n_calls, verbose=True)
    print('*' * 30)
    print('best result:')
    print('f1:', -result.fun)
    space_dim_name = [item.name for item in search_space]
    for hyper, value in zip(space_dim_name, result.x):
        print(hyper, value)


if __name__ == "__main__":
    optimize(12)
