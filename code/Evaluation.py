import torch
import dill
import os
import numpy as np
import pandas as pd

from torch.autograd import Variable
from tqdm import tqdm

from Networks import EncoderSeq, DecoderSeqAttnDe
from Parameters import Params

params = Params()


# evaluate the seq2seq model
class EvaluateSeq2Seq:
    def __init__(self, device, concept2id_file, patient_records_file, predict_max_length, encoder_output_max_length):
        self.device = device
        self.concept2id_file = concept2id_file
        self.patient_records_file = patient_records_file
        self.predict_max_length = predict_max_length
        self.encoder_output_max_length = encoder_output_max_length

        voc = dill.load(open(self.concept2id_file, 'rb'))
        self.diag_voc = voc['diag_voc']
        self.pro_voc = voc['pro_voc']
        self.med_voc = voc['med_voc']

        self.diagnose_count = len(self.diag_voc.word2idx)
        self.procedure_count = len(self.pro_voc.word2idx)
        self.medication_count = len(self.med_voc.word2idx)

        self.evaluate_utils = EvaluationUtil()

    def metric_jaccard_similarity(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_jaccard_similarity(predict_medications, target_medications)

    def metric_precision(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_precision(predict_medications, target_medications)

    def metric_recall(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_recall(predict_medications, target_medications)

    def metric_f1(self, precision, recall):
        return self.evaluate_utils.metric_f1(precision, recall)

    # evaluate the Delete mode
    def evaluateItersDelete(self, encoder, decoder, patient_records, save_result_file=None):
        total_metric_jaccard = 0.0
        total_metric_precision = 0.0
        total_metric_recall = 0.0
        count = 0
        predict_result_patient_records = []

        for i, patient in enumerate(tqdm(patient_records)):
            current_patient = []
            for idx, adm in enumerate(patient):
                count += 1
                current_records = patient[:idx + 1]
                target_medication = adm[3]

                query, encoder_output_tmp = encoder(current_records)
                encoder_output = torch.zeros(self.encoder_output_max_length, encoder.hidden_size,
                                             device=self.device)
                if encoder_output_tmp.size(0) <= self.encoder_output_max_length:
                    encoder_output[:encoder_output_tmp.size(0)] = encoder_output_tmp
                else:
                    encoder_output = encoder_output_tmp[-self.encoder_output_max_length:]

                decoder_input = torch.LongTensor([[params.SOS_id]]).to(self.device)
                decoder_hidden = query.unsqueeze(dim=1)
                predict_token = []
                predict_prob = []
                for di in range(self.predict_max_length):
                    decoder_output, decoder_hidden, output_prob = decoder(decoder_input, decoder_hidden, encoder_output)
                    topv, topi = decoder_output.data.topk(self.medication_count)
                    predict_prob.append(output_prob[0])

                    ni = topi.squeeze().detach().cpu().numpy()
                    for token in ni:
                        if token not in predict_token:
                            next_token = token
                            break

                    if next_token == params.EOS_id:
                        break
                    predict_token.append(next_token)
                    decoder_input = torch.LongTensor([[next_token]]).to(self.device)

                predict_token = [item for item in predict_token if item != params.EOS_id]

                jaccard = self.evaluate_utils.metric_jaccard_similarity(predict_token, target_medication)
                precision = self.evaluate_utils.metric_precision(predict_token, target_medication)
                recall = self.evaluate_utils.metric_recall(predict_token, target_medication)
                f1 = self.evaluate_utils.metric_f1(precision, recall)

                target_multi_hot = np.zeros(self.medication_count - 2)
                target_index = [item - 2 for item in target_medication]
                target_multi_hot[target_index] = 1
                predict_prob = np.mean(np.array(predict_prob)[:, 2:], axis=0)

                total_metric_jaccard += jaccard
                total_metric_precision += precision
                total_metric_recall += recall

                adm.append(predict_token)
                current_patient.append(adm)

            predict_result_patient_records.append(current_patient)

        jaccard_avg = total_metric_jaccard / count
        precision_avg = total_metric_precision / count
        recall_avg = total_metric_recall / count
        f1_avg = self.evaluate_utils.metric_f1(precision_avg, recall_avg)

        dill.dump(obj=predict_result_patient_records,
                  file=open(os.path.join(save_result_file, 'predict_result.pkl'), 'wb'))

        print('evaluation result:')
        print('  jaccard:', jaccard_avg)
        print('precision:', precision_avg)
        print('   recall:', recall_avg)
        print('       f1:', f1_avg)

    # evaluate the Greedy search mode
    def evaluateItersGreedy(self, encoder, decoder, patient_records, save_result_file=None):
        total_metric_jaccard = 0.0
        total_metric_precision = 0.0
        total_metric_recall = 0.0
        count = 0

        predict_distinct_rate = []

        predict_result_patient_records = []

        for i, patient in enumerate(tqdm(patient_records)):
            current_patient = []
            for idx, adm in enumerate(patient):
                count += 1
                current_records = patient[:idx + 1]
                target_medication = adm[3]

                query, encoder_output_tmp = encoder(current_records)
                encoder_output = torch.zeros(self.encoder_output_max_length, encoder.hidden_size,
                                             device=self.device)
                if encoder_output_tmp.size(0) <= self.encoder_output_max_length:
                    encoder_output[:encoder_output_tmp.size(0)] = encoder_output_tmp
                else:
                    encoder_output = encoder_output_tmp[-self.encoder_output_max_length:]

                decoder_input = torch.LongTensor([[params.SOS_id]]).to(self.device)
                decoder_hidden = query.unsqueeze(dim=1)
                predict_token = []
                predict_prob = []

                for di in range(self.predict_max_length):
                    decoder_output, decoder_hidden, output_prob = decoder(decoder_input, decoder_hidden, encoder_output)
                    topv, topi = decoder_output.data.topk(1)
                    predict_prob.append(output_prob[0])
                    if topi.item() == params.EOS_id:
                        break
                    else:
                        predict_token.append(topi.item())
                    decoder_input = topi.squeeze().detach()

                predict_token = [item for item in predict_token if item != params.EOS_id]

                jaccard = self.evaluate_utils.metric_jaccard_similarity(predict_token, target_medication)
                precision = self.evaluate_utils.metric_precision(predict_token, target_medication)
                recall = self.evaluate_utils.metric_recall(predict_token, target_medication)
                f1 = self.evaluate_utils.metric_f1(precision, recall)

                target_multi_hot = np.zeros(self.medication_count - 2)
                target_index = [item - 2 for item in target_medication]
                target_multi_hot[target_index] = 1
                predict_prob = np.mean(np.array(predict_prob)[:, 2:], axis=0)

                total_metric_jaccard += jaccard
                total_metric_precision += precision
                total_metric_recall += recall

                adm.append(predict_token)
                current_patient.append(adm)
                predict_distinct_rate.append(float(len(set(predict_token)) / len(predict_token)))

            predict_result_patient_records.append(current_patient)

        jaccard_avg = total_metric_jaccard / count
        precision_avg = total_metric_precision / count
        recall_avg = total_metric_recall / count
        f1_avg = self.evaluate_utils.metric_f1(precision_avg, recall_avg)

        dill.dump(obj=predict_result_patient_records,
                  file=open(os.path.join(save_result_file, 'predict_result.pkl'), 'wb'))

        print('evaluation result:')
        print('  jaccard:', jaccard_avg)
        print('precision:', precision_avg)
        print('   recall:', recall_avg)
        print('       f1:', f1_avg)

    def evaluate(self, load_model_name, input_size, hidden_size, encoder_n_layers, encoder_input_embedding_dropout_rate,
                 encoder_gru_dropout_rate, decoder_input_embedding_dropout_rate, decoding_method='standard',
                 save_result_path=None):
        print('load model from checkpoint file:', load_model_name)
        checkpoint = torch.load(load_model_name)

        print('build encoder and decoder >>>')
        encoder = EncoderSeq(self.device, input_size, hidden_size, self.diagnose_count, self.procedure_count,
                             encoder_n_layers, encoder_input_embedding_dropout_rate, encoder_gru_dropout_rate, False)
        decoder = DecoderSeqAttnDe(hidden_size, self.medication_count, decoder_input_embedding_dropout_rate,
                                   self.encoder_output_max_length, self.device)

        if decoding_method != 'greedy' and decoding_method != 'delete':
            print('wrong decoding method, choose from greedy and delete')

        encoder_sd = checkpoint['encoder']
        decoder_sd = checkpoint['decoder']
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        encoder.eval()
        decoder.eval()

        patient_records = pd.read_pickle(self.patient_records_file)
        split_point = int(len(patient_records) * params.train_ratio)
        test_count = int(len(patient_records) * params.test_ratio)
        patient_records_train = patient_records[:split_point]
        patient_records_test = patient_records[split_point:split_point + test_count]
        patient_records_validation = patient_records[split_point + test_count:split_point + test_count + test_count]

        model_parameter = load_model_name.split('/')[-4:-1]
        save_result_file = '/'
        save_result_file = save_result_file.join(model_parameter)
        save_result_path = os.path.join(save_result_path, save_result_file)

        if not os.path.exists(save_result_path):
            os.makedirs(save_result_path)

        print('start evaluating >>>')
        if decoding_method == 'greedy':
            self.evaluateItersGreedy(encoder, decoder, patient_records_validation, save_result_path)
        elif decoding_method == 'delete':
            self.evaluateItersDelete(encoder, decoder, patient_records_validation, save_result_path)
        else:
            print('wrong evaluation type, choose from standard and delete')


# occurrence-based beam search
class BeamOccur(object):
    def __init__(self, beam_size, minus_weight, SOS_id=params.SOS_id, EOS_id=params.EOS_id,
                 medication_count=params.MEDICATION_COUNT, device=torch.device('cpu:0')):
        self.beam_size = beam_size
        self.SOS_id = SOS_id
        self.EOS_id = EOS_id
        self.medication_count = medication_count
        self.device = device
        self.done = False
        self.minus_weight = minus_weight
        # dim=(beam_size, medication_count)
        self.weight_matrix = torch.mul(torch.ones(self.beam_size, self.medication_count), self.minus_weight).to(device)

        # the score for eacm translation on the beam, size=([beam_size])
        self.scores = torch.FloatTensor(self.beam_size).zero_().to(self.device)
        # the backpointers at each time step, indicate the index in the last step
        self.prevKs = []
        # the outputs at each time step, also the inputs of the next step
        self.nextYs = [torch.LongTensor(1).fill_(self.SOS_id).to(self.device)]
        # historical #occurrence of medications of each translation, one translation for a row
        self.occurrence = torch.zeros(self.beam_size, self.medication_count).to(self.device)

    # get the outputs for the current time step
    def get_current_state(self):
        return self.nextYs[-1]

    # get the backpointers for the current time step
    def get_current_origin(self):
        return self.prevKs[-1]

    def advance(self, log_probs, probs):
        """
        :param log_probs: log probabilities of advancing from last step, dim=(1 x medication_count) or (beam_size, medication_count)
        :return: true if beam search is done
        """
        medication_count = log_probs.size(1)
        if len(self.prevKs) > 0:
            # dim=(beam_size, medication_count)
            beam_lk = log_probs + self.scores.unsqueeze(1).expand_as(log_probs) - torch.mul(self.occurrence,
                                                                                            self.weight_matrix)
        else:
            beam_lk = log_probs  # dim=(1,medication_count)
        flat_beam_lk = beam_lk.view(-1)  # dim=(medication_count) or (beam_size x medication_count)

        best_scores, best_id = flat_beam_lk.topk(self.beam_size, 0, True, True)
        self.scores = best_scores  # dim=(beam_size)
        prev_k = best_id / medication_count

        if len(self.prevKs) > 0:
            self.occurrence.data.copy_(self.occurrence.data.index_select(0, prev_k))
            self.occurrence[torch.arange(0, self.beam_size), best_id - prev_k * medication_count] += 1
        else:
            self.occurrence[torch.arange(0, self.beam_size), best_id - prev_k * medication_count] += 1

        self.prevKs.append(prev_k)
        self.nextYs.append(best_id - prev_k * medication_count)

        if self.nextYs[-1][0] == self.EOS_id:
            self.done = True

        return self.done

    def sort_best(self):
        return torch.sort(self.scores, 0, True)

    def get_hyp(self, k):
        hyp = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
        return hyp[::-1]  # reverse the list


# standard Beam search
class Beam(object):
    def __init__(self, beam_size, SOS_id=params.SOS_id, EOS_id=params.EOS_id, medication_count=params.MEDICATION_COUNT,
                 device=torch.device('cpu:0')):
        self.beam_size = beam_size
        self.SOS_id = SOS_id
        self.EOS_id = EOS_id
        self.medication_count = medication_count
        self.device = device
        self.done = False

        # the score for eacm translation on the beam, size=([beam_size])
        self.scores = torch.FloatTensor(self.beam_size).zero_().to(self.device)
        # the backpointers at each time step, indicate the index in the last step
        self.prevKs = []
        # the outputs at each time step, also the inputs of the next step
        self.nextYs = [torch.LongTensor(1).fill_(self.SOS_id).to(self.device)]

    # get the outputs for the current time step
    def get_current_state(self):
        return self.nextYs[-1]

    # get the backpointers for the current time step
    def get_current_origin(self):
        return self.prevKs[-1]

    def advance(self, log_probs, probs):
        """
        :param probability: probabilities of advancing from las step, dim=(1 x medication_count) or (beam_size, medication_count)
        :return: true if beam search is done
        """
        medication_count = log_probs.size(1)
        if len(self.prevKs) > 0:
            beam_lk = log_probs + self.scores.unsqueeze(1).expand_as(log_probs)  # dim=(beam_size, medication_count)
        else:
            beam_lk = log_probs  # dim=(1,medication_count)
        flat_beam_lk = beam_lk.view(-1)  # dim=(medication_count) or (beam_size x medication_count)

        best_scores, best_id = flat_beam_lk.topk(self.beam_size, 0, True, True)
        self.scores = best_scores  # dim=(beam_size)
        prev_k = best_id / medication_count

        self.prevKs.append(prev_k)
        self.nextYs.append(best_id - prev_k * medication_count)

        if self.nextYs[-1][0] == self.EOS_id:
            self.done = True

        return self.done

    def sort_best(self):
        return torch.sort(self.scores, 0, True)

    def get_hyp(self, k):
        hyp = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
        return hyp[::-1]  # reverse the list


class EvaluateSeq2SeqBeamSearch:
    def __init__(self, device, concept2id_file, patient_records_file, predict_max_length, encoder_output_max_length,
                 beam_minus_weight=0.1, beam_type='occurrence', beam_size=5, n_best=1):
        self.device = device
        self.concept2id_file = concept2id_file
        self.patient_records_file = patient_records_file
        self.predict_max_length = predict_max_length
        self.encoder_output_max_length = encoder_output_max_length
        self.beam_minus_weight = beam_minus_weight

        if beam_type != 'standard' and beam_type != 'occurrence':
            print('wrong beam type, choose from standard and occurrence')
            return
        self.beam_type = beam_type
        self.beam_size = beam_size
        self.n_best = n_best

        voc = dill.load(open(self.concept2id_file, 'rb'))
        self.diag_voc = voc['diag_voc']
        self.pro_voc = voc['pro_voc']
        self.med_voc = voc['med_voc']

        self.diagnose_count = len(self.diag_voc.word2idx)
        self.procedure_count = len(self.pro_voc.word2idx)
        self.medication_count = len(self.med_voc.word2idx)

        self.evaluate_utils = EvaluationUtil()

    def metric_jaccard_similarity(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_jaccard_similarity(predict_medications, target_medications)

    def metric_precision(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_precision(predict_medications, target_medications)

    def metric_recall(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_recall(predict_medications, target_medications)

    def metric_f1(self, precision, recall):
        return self.evaluate_utils.metric_f1(precision, recall)

    def evaluateIters(self, encoder, decoder, patient_records, save_result_file=None):
        total_metric_jaccard = 0.0
        total_metric_precision = 0.0
        total_metric_recall = 0.0
        count = 0
        beam_size = self.beam_size

        predict_result_patient_records = []

        for i, patient in enumerate(tqdm(patient_records)):
            current_patient = []
            for idx, adm in enumerate(patient):
                count += 1
                current_records = patient[:idx + 1]
                target_medication = adm[3]
                # dim(query)=(1,hidden_size),dim(encoder_output)=(encoder_output_max_length,hidden_size)
                query, encoder_output_tmp = encoder(current_records)
                encoder_output = torch.zeros(self.encoder_output_max_length, encoder.hidden_size,
                                             device=self.device)
                if encoder_output_tmp.size(0) <= self.encoder_output_max_length:
                    encoder_output[:encoder_output_tmp.size(0)] = encoder_output_tmp
                else:
                    encoder_output = encoder_output_tmp[-self.encoder_output_max_length:]

                if self.beam_type == 'standard':
                    beam = Beam(beam_size, params.SOS_id, params.EOS_id, self.medication_count, self.device)
                else:
                    beam = BeamOccur(beam_size, self.beam_minus_weight, params.SOS_id, params.EOS_id,
                                     self.medication_count, self.device)
                decoder_input = beam.get_current_state().view(1, -1).to(self.device)  # dim=(1,1)
                decoder_hidden = Variable(query.unsqueeze(1).data)  # dim=(1,1,hidden_size)

                # dim(decoder_output)=(1,medication_count), dim(decoder_hidden)=(1,1,hidden_size)
                decoder_output, decoder_hidden, output_probs = decoder(decoder_input, decoder_hidden, encoder_output)
                output_probs = torch.from_numpy(output_probs)

                if not beam.advance(decoder_output, output_probs):
                    decoder_hidden = Variable(decoder_hidden.data.repeat(1, beam_size, 1))  # dim=(1,beam_size,hidden)
                    for _ in range(self.predict_max_length - 1):
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

                        log_probs = torch.stack(new_output).squeeze(1)  # dim=(beam_size,medication_count)
                        decoder_hidden = torch.stack(new_hidden).unsqueeze(0)  # dim=(1,beam_size,hidden_size)
                        probs = np.stack(new_probs)
                        probs = torch.from_numpy(probs).to(self.device)  # dim=(beam_size,medication_count)
                        if beam.advance(log_probs, probs):
                            break

                scores, ks = beam.sort_best()
                scores = scores[:self.n_best]
                hyps = [beam.get_hyp(k) for k in ks[:self.n_best]]

                hyp = hyps[0]
                predict_token = torch.stack(hyp).detach().cpu().numpy()

                predict_token = [item for item in predict_token if item != params.EOS_id]
                jaccard = self.evaluate_utils.metric_jaccard_similarity(predict_token, target_medication)
                precision = self.evaluate_utils.metric_precision(predict_token, target_medication)
                recall = self.evaluate_utils.metric_recall(predict_token, target_medication)
                f1 = self.evaluate_utils.metric_f1(precision, recall)

                target_multi_hot = np.zeros(self.medication_count - 2)
                target_index = [item - 2 for item in target_medication]
                target_multi_hot[target_index] = 1

                total_metric_jaccard += jaccard
                total_metric_precision += precision
                total_metric_recall += recall

                adm.append(predict_token)
                current_patient.append(adm)

            predict_result_patient_records.append(current_patient)

        jaccard_avg = total_metric_jaccard / count
        precision_avg = total_metric_precision / count
        recall_avg = total_metric_recall / count
        f1_avg = self.metric_f1(precision_avg, recall_avg)

        dill.dump(obj=predict_result_patient_records,
                  file=open(os.path.join(save_result_file, 'predict_result.pkl'), 'wb'))

        print('evaluation result:')
        print('  jaccard:', jaccard_avg)
        print('precision:', precision_avg)
        print('   recall:', recall_avg)
        print('       f1:', f1_avg)

    def evaluate(self, load_model_name, input_size, hidden_size, encoder_n_layers, encoder_input_embedding_dropout_rate,
                 encoder_gru_dropout_rate, decoder_input_embedding_dropout_rate, save_result_path=None):
        print('load model from checkpoint file:', load_model_name)
        checkpoint = torch.load(load_model_name)

        print('build encoder and decoder >>>')
        encoder = EncoderSeq(self.device, input_size, hidden_size, self.diagnose_count, self.procedure_count,
                             encoder_n_layers, encoder_input_embedding_dropout_rate, encoder_gru_dropout_rate, False)
        decoder = DecoderSeqAttnDe(hidden_size, self.medication_count, decoder_input_embedding_dropout_rate,
                                   self.encoder_output_max_length, self.device)

        encoder_sd = checkpoint['encoder']
        decoder_sd = checkpoint['decoder']
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        encoder.eval()
        decoder.eval()

        patient_records = pd.read_pickle(self.patient_records_file)
        split_point = int(len(patient_records) * params.train_ratio)
        test_count = int(len(patient_records) * params.test_ratio)
        patient_records_train = patient_records[:split_point]
        patient_records_test = patient_records[split_point:split_point + test_count]
        patient_records_validation = patient_records[split_point + test_count:split_point + test_count + test_count]

        model_parameter = load_model_name.split('/')[-4:-1]
        save_result_file = '/'
        save_result_file = save_result_file.join(model_parameter)
        save_result_path = os.path.join(save_result_path, save_result_file)

        if not os.path.exists(save_result_path):
            os.makedirs(save_result_path)

        self.evaluateIters(encoder, decoder, patient_records_validation, save_result_path)


class EvaluationUtil:
    def metric_jaccard_similarity(self, predict_prescriptions, target_prescriptions):
        union = list(set(predict_prescriptions) | set(target_prescriptions))
        intersection = list(set(predict_prescriptions) & set(target_prescriptions))
        jaccard = float(len(intersection)) / len(union)
        return jaccard

    def metric_precision(self, predict_prescriptions, target_prescriptions):
        if len(set(predict_prescriptions)) == 0:
            return 0
        intersection = list(set(predict_prescriptions) & set(target_prescriptions))
        # precision = float(len(intersection)) / len(set(predict_prescriptions))
        precision = float(len(intersection)) / len(predict_prescriptions)
        return precision

    def metric_recall(self, predict_prescriptions, target_prescriptions):
        intersection = list(set(predict_prescriptions) & set(target_prescriptions))
        # recall = float(len(intersection)) / len(set(target_prescriptions))
        recall = float(len(intersection)) / len(target_prescriptions)
        return recall

    def metric_f1(self, precision, recall):
        if precision + recall == 0:
            return 0
        f1 = 2.0 * precision * recall / (precision + recall)
        return f1


