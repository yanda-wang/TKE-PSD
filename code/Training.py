import torch
import os
import datetime
import pickle
import dill
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim

from Networks import EncoderLinearQuery, DecoderKeyValueGCNMultiEmbedding
from Networks import EncoderSeq, DecoderSeqAttnDe
from Parameters import Params
from Evaluation import EvaluationUtil

params = Params()


class TrainMedRecSeq2Set:
    def __init__(self, device, patient_records_file, voc_file, ehr_matrix_file):
        self.device = device
        self.patient_records_file = patient_records_file
        self.voc_file = voc_file
        self.ehr_matrix_file = ehr_matrix_file

        voc = dill.load(open(self.voc_file, 'rb'))
        self.diag_voc = voc['diag_voc']
        self.pro_voc = voc['pro_voc']
        self.med_voc = voc['med_voc']

        self.diagnose_count = len(self.diag_voc.word2idx)
        self.procedure_count = len(self.pro_voc.word2idx)
        self.medication_count = len(self.med_voc.word2idx)

        self.ehr_matrix = dill.load(open(self.ehr_matrix_file, 'rb'))
        self.evaluate_utils = EvaluationUtil()

    def loss_function(self, target_medications, predict_medications, proportion_bce, proportion_multi):
        loss_bce_target = np.zeros((1, self.medication_count))
        loss_bce_target[:, target_medications] = 1
        loss_multi_target = np.full((1, self.medication_count), -1)
        for idx, item in enumerate(target_medications):
            loss_multi_target[0][idx] = item

        loss_bce = F.binary_cross_entropy_with_logits(predict_medications,
                                                      torch.FloatTensor(loss_bce_target).to(self.device))
        loss_multi = F.multilabel_margin_loss(torch.sigmoid(predict_medications),
                                              torch.LongTensor(loss_multi_target).to(self.device))
        loss = proportion_bce * loss_bce + proportion_multi * loss_multi
        return loss

    def get_performance_on_testset(self, encoder, decoder, patient_records, proportion_bce=0.9, proportion_multi=0.1):
        jaccard_avg, precision_avg, recall_avg, loss_avg = 0.0, 0.0, 0.0, 0.0
        count = 0
        for patient in patient_records:
            for idx, adm in enumerate(patient):
                count += 1
                current_records = patient[:idx + 1]
                target_medications = adm[2]
                query, memory_keys, memory_values = encoder(current_records)
                predict_output, _ = decoder(query, memory_keys, memory_values)

                target_multi_hot = np.zeros(self.medication_count)
                target_multi_hot[target_medications] = 1
                predict_prob = torch.sigmoid(predict_output).detach().cpu().numpy()[0]
                predict_multi_hot = predict_prob.copy()
                index_nan = np.argwhere(np.isnan(predict_multi_hot))
                if index_nan.shape[0] != 0:
                    predict_multi_hot = np.zeros_like(predict_multi_hot)

                predict_multi_hot[predict_multi_hot >= 0.5] = 1
                predict_multi_hot[predict_multi_hot < 0.5] = 0
                predict_medications = list(np.where(predict_multi_hot == 1)[0])

                jaccard = self.evaluate_utils.metric_jaccard_similarity(predict_medications, target_medications)
                precision = self.evaluate_utils.metric_precision(predict_medications, target_medications)
                recall = self.evaluate_utils.metric_recall(predict_medications, target_medications)
                loss = self.loss_function(target_medications, predict_output, proportion_bce, proportion_multi)

                jaccard_avg += jaccard
                precision_avg += precision
                recall_avg += recall
                loss_avg += loss.item()

        jaccard_avg = jaccard_avg / count
        precision_avg = precision_avg / count
        recall_avg = recall_avg / count
        f1_avg = self.evaluate_utils.metric_f1(precision_avg, recall_avg)
        loss_avg = loss_avg / count

        return jaccard_avg, precision_avg, recall_avg, f1_avg, loss_avg

    def trainIters(self, encoder, decoder, encoder_optimizer, decoder_optimizer, patient_records_train,
                   patient_records_test, save_model_path, n_epoch, print_every_iteration=100, save_every_epoch=5,
                   proportion_bce=0.9, proportion_multi=0.1, trained_epoch=0, trained_iteration=0):

        start_epoch = trained_epoch + 1
        trained_n_iteration = trained_iteration

        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        log_file = open(os.path.join(save_model_path, 'medrec_loss.log'), 'a+')

        encoder_lr_scheduler = ReduceLROnPlateau(encoder_optimizer, patience=10, factor=0.1)
        decoder_lr_scheduler = ReduceLROnPlateau(decoder_optimizer, patience=10, factor=0.1)

        for epoch in range(start_epoch, start_epoch + n_epoch):
            print_loss = 0
            iteration = 0
            for patient in patient_records_train:
                for idx, adm in enumerate(patient):
                    trained_n_iteration += 1
                    iteration += 1
                    current_records = patient[:idx + 1]
                    target_medications = adm[2]
                    encoder_optimizer.zero_grad()
                    decoder_optimizer.zero_grad()
                    query, memory_keys, memory_values = encoder(current_records)
                    predict_output, _ = decoder(query, memory_keys, memory_values)
                    loss = self.loss_function(target_medications, predict_output, proportion_bce, proportion_multi)
                    print_loss += loss.item()
                    loss.backward()
                    encoder_optimizer.step()
                    decoder_optimizer.step()

                    if iteration % print_every_iteration == 0:
                        print_loss_avg = print_loss / print_every_iteration
                        print_loss = 0.0
                        print(
                            'epoch: {}; time: {}; Iteration: {}; train loss: {:.4f}'.format(
                                epoch, datetime.datetime.now(), trained_n_iteration, print_loss_avg))
                        log_file.write(
                            'epoch: {}; time: {}; Iteration: {};  train loss: {:.4f}\n'.format(
                                epoch, datetime.datetime.now(), trained_n_iteration, print_loss_avg))

            encoder.eval()
            decoder.eval()
            jaccard_avg, precision_avg, recall_avg, f1_avg, print_loss_avg_on_test = self.get_performance_on_testset(
                encoder, decoder, patient_records_test, proportion_bce, proportion_multi)
            encoder.train()
            decoder.train()

            print(
                'epoch: {}; time: {}; Iteration: {}; train loss: {:.4f}; test loss: {:.4f}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}'.format(
                    epoch, datetime.datetime.now(), trained_n_iteration, print_loss_avg, print_loss_avg_on_test,
                    jaccard_avg, precision_avg, recall_avg, f1_avg))
            log_file.write(
                'epoch: {}; time: {}; Iteration: {}; train loss: {:.4f}; test loss: {:.4f}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}\n'.format(
                    epoch, datetime.datetime.now(), trained_n_iteration, print_loss_avg, print_loss_avg_on_test,
                    jaccard_avg, precision_avg, recall_avg, f1_avg))

            encoder_lr_scheduler.step(print_loss_avg)
            decoder_lr_scheduler.step(print_loss_avg)

            if epoch % save_every_epoch == 0:
                torch.save(
                    {'medrec_epoch': epoch,
                     'medrec_iteration': trained_n_iteration,
                     'encoder': encoder.state_dict(),
                     'decoder': decoder.state_dict(),
                     'encoder_optimizer': encoder_optimizer.state_dict(),
                     'decoder_optimizer': decoder_optimizer.state_dict(),
                     'medrec_avg_loss_train': print_loss_avg,
                     'medrec_avg_loss_test': print_loss_avg_on_test},
                    os.path.join(save_model_path,
                                 'medrec_{}_{}_{:.4f}_{:.4f}_{:.4f}.checkpoint'.format(epoch, trained_n_iteration,
                                                                                       print_loss_avg,
                                                                                       print_loss_avg_on_test, f1_avg)))

        log_file.close()

    def train(self, input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate,
              encoder_bidirectional, encoder_regular_lambda, encoder_learning_rate, decoder_dropout_rate,
              decoder_regular_lambda, decoder_learning_rate, hop, attn_type_kv, attn_type_embedding,
              save_model_dir, proportion_bce, proportion_multi,
              n_epoch=50, print_every_iteration=100, save_every_epoch=1, load_model_name=None):

        print('initializing >>>')
        if load_model_name:
            print('load model from checkpoint file: ', load_model_name)
            checkpoint = torch.load(load_model_name)

        print('build medrec model >>>')

        encoder = EncoderLinearQuery(self.device, input_size, hidden_size, self.diagnose_count, self.procedure_count,
                                     encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate,
                                     encoder_bidirectional)
        decoder = DecoderKeyValueGCNMultiEmbedding(self.device, hidden_size, self.medication_count,
                                                   self.medication_count, hop, decoder_dropout_rate, attn_type_kv,
                                                   attn_type_embedding, self.ehr_matrix)
        if load_model_name:
            encoder_sd = checkpoint['encoder']
            decoder_sd = checkpoint['decoder']
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        encoder.train()
        decoder.train()

        print('build optimizer >>>')
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_learning_rate,
                                       weight_decay=encoder_regular_lambda)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_learning_rate,
                                       weight_decay=decoder_regular_lambda)
        if load_model_name:
            encoder_optimizer_sd = checkpoint['encoder_optimizer']
            decoder_optimizer_sd = checkpoint['decoder_optimizer']
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        print('start training medrec model >>>')
        patient_records = pd.read_pickle(self.patient_records_file)
        split_point = int(len(patient_records) * params.train_ratio)
        test_count = int(len(patient_records) * params.test_ratio)
        patient_records_train = patient_records[:split_point]
        patient_records_test = patient_records[split_point:split_point + test_count]

        medrec_trained_epoch = 0
        medrec_trained_iteration = 0

        if load_model_name:
            medrec_trained_n_epoch_sd = checkpoint['medrec_epoch']
            medrec_trained_n_iteration_sd = checkpoint['medrec_iteration']
            medrec_trained_epoch = medrec_trained_n_epoch_sd
            medrec_trained_iteration = medrec_trained_n_iteration_sd

        # set save_model_path
        save_model_structure = str(encoder_n_layers) + '_' + str(input_size) + '_' + str(hidden_size) + '_' + str(
            encoder_bidirectional) + '_' + str(attn_type_kv) + '_' + str(attn_type_embedding)
        save_model_parameters = str(encoder_embedding_dropout_rate) + '_' + str(
            encoder_gru_dropout_rate) + '_' + str(encoder_regular_lambda) + '_' + str(
            encoder_learning_rate) + '_' + str(decoder_dropout_rate) + '_' + str(
            decoder_regular_lambda) + '_' + str(decoder_learning_rate) + '_' + str(hop)
        save_model_path = os.path.join(save_model_dir, save_model_structure, save_model_parameters)

        self.trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, patient_records_train,
                        patient_records_test, save_model_path, n_epoch, print_every_iteration, save_every_epoch,
                        proportion_bce, proportion_multi, medrec_trained_epoch, medrec_trained_iteration)


class TrainMedRecSeq2Seq:
    def __init__(self, device, patient_records_file, voc_file, predict_max_length, encoder_hidden_max_length):
        self.device = device
        self.patient_records_file = patient_records_file
        self.voc_file = voc_file
        self.predict_max_length = predict_max_length
        self.encoder_hidden_max_length = encoder_hidden_max_length

        voc = dill.load(open(self.voc_file, 'rb'))
        self.diag_voc = voc['diag_voc']
        self.pro_voc = voc['pro_voc']
        self.med_voc = voc['med_voc']

        self.diagnose_count = len(self.diag_voc.word2idx)
        self.procedure_count = len(self.pro_voc.word2idx)
        self.medication_count = len(self.med_voc.word2idx)

        # self.ehr_matrix = dill.load(open(self.ehr_matrix_file, 'rb'))
        self.evaluate_utils = EvaluationUtil()

    def get_performance_on_testset(self, encoder, decoder, patient_records):
        jaccard_avg, precision_avg, recall_avg = 0.0, 0.0, 0.0
        count = 0
        criterion = nn.NLLLoss()
        for patient in patient_records:
            for idx, adm in enumerate(patient):
                count += 1
                current_records = patient[:idx + 1]
                y_predict_token = []
                y_predict_prob = []
                target = adm[3]

                query, encoder_output_tmp = encoder(current_records)
                encoder_output = torch.zeros(self.encoder_hidden_max_length, encoder.hidden_size, device=self.device)
                if encoder_output_tmp.size(0) <= self.encoder_hidden_max_length:
                    encoder_output[:encoder_output_tmp.size(0)] = encoder_output_tmp
                else:
                    encoder_output = encoder_output_tmp[-self.encoder_hidden_max_length:]
                decoder_input = torch.LongTensor([[params.SOS_id]]).to(self.device)
                decoder_hidden = query.unsqueeze(dim=1)

                for di in range(self.predict_max_length):
                    decoder_output, decoder_hidden, output_prob = decoder(decoder_input, decoder_hidden, encoder_output)
                    topv, topi = decoder_output.data.topk(1)
                    decoder_input = topi.squeeze().detach()
                    y_predict_token.append(decoder_input.item())
                    y_predict_prob.append(output_prob[0])
                    if decoder_input.item() == params.EOS_id:
                        y_predict_token = y_predict_token[:-1]
                        y_predict_prob = y_predict_prob[:-1]
                        break

                jaccard = self.evaluate_utils.metric_jaccard_similarity(y_predict_token, target)
                precision = self.evaluate_utils.metric_precision(y_predict_token, target)
                recall = self.evaluate_utils.metric_recall(y_predict_token, target)

                target_multi_hot = np.zeros(self.medication_count - 2)
                target_index = [item - 2 for item in target]
                target_multi_hot[target_index] = 1
                predict_prob = np.mean(np.array(y_predict_prob)[:, 2:], axis=0)

                jaccard_avg += jaccard
                precision_avg += precision
                recall_avg += recall

        jaccard_avg = jaccard_avg / count
        precision_avg = precision_avg / count
        recall_avg = recall_avg / count
        f1_avg = self.evaluate_utils.metric_f1(precision_avg, recall_avg)
        return jaccard_avg, precision_avg, recall_avg, f1_avg

    def trainIters(self, encoder, decoder, encoder_optimizer, decoder_optimizer, teaching_force_rate,
                   patient_records_train, patient_records_test, save_model_path, n_epoch, print_every_iteration=100,
                   save_every_epoch=1, trained_epoch=0, trained_iteration=0):
        start_epoch = trained_epoch + 1
        trained_n_iteration = trained_iteration
        criterion = nn.NLLLoss()

        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        log_file = open(os.path.join(save_model_path, 'seq2seq_loss.log'), 'a+')

        for epoch in range(start_epoch, start_epoch + n_epoch):
            if epoch >= 40 and epoch % 10 == 0:
                teaching_force_rate -= 0.1
            if teaching_force_rate < 0:
                teaching_force_rate = 0

            print_loss = 0.0
            iteration = 0
            for patient in patient_records_train:
                for idx, adm in enumerate(patient):
                    loss = 0.0
                    trained_n_iteration += 1
                    iteration += 1
                    current_records = patient[:idx + 1]

                    target_medications = adm[3] + [params.EOS_id]
                    target_length = len(target_medications)
                    target_medications = torch.LongTensor(target_medications).to(self.device).view(-1, 1)
                    encoder_optimizer.zero_grad()
                    decoder_optimizer.zero_grad()
                    query, encoder_output_tmp = encoder(current_records)

                    encoder_output = torch.zeros(self.encoder_hidden_max_length, encoder.hidden_size,
                                                 device=self.device)
                    if encoder_output_tmp.size(0) <= self.encoder_hidden_max_length:
                        encoder_output[:encoder_output_tmp.size(0)] = encoder_output_tmp
                    else:
                        encoder_output = encoder_output_tmp[-self.encoder_hidden_max_length:]
                    decoder_input = torch.LongTensor([[params.SOS_id]]).to(self.device)
                    decoder_hidden = query.unsqueeze(dim=1)
                    use_teaching_force = True if random.random() < teaching_force_rate else False

                    if use_teaching_force:
                        for di in range(target_length):
                            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
                            loss += criterion(decoder_output, target_medications[di])
                            decoder_input = target_medications[di]
                    else:
                        for di in range(target_length):
                            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
                            loss += criterion(decoder_output, target_medications[di])

                            topv, topi = decoder_output.topk(1)
                            decoder_input = topi.squeeze().detach()
                            if decoder_input.item() == params.EOS_id:
                                break
                    loss.backward()
                    encoder_optimizer.step()
                    decoder_optimizer.step()
                    print_loss += loss.item() / target_length

                    if iteration % print_every_iteration == 0:
                        print_loss_avg = print_loss / print_every_iteration
                        print_loss = 0
                        print(
                            'epoch: {}; time: {}; Iteration: {}; train loss: {:.4f}'.format(
                                epoch, datetime.datetime.now(), trained_n_iteration, print_loss_avg))
                        log_file.write(
                            'epoch: {}; time: {}; Iteration: {};  train loss: {:.4f}\n'.format(
                                epoch, datetime.datetime.now(), trained_n_iteration, print_loss_avg))

            encoder.eval()
            decoder.eval()
            jaccard_avg, precision_avg, recall_avg, f1_avg = self.get_performance_on_testset(encoder, decoder,
                                                                                             patient_records_test)
            encoder.train()
            decoder.train()
            print(
                'epoch: {}; time: {}; Iteration: {}; train loss: {:.4f}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}'.format(
                    epoch, datetime.datetime.now(), trained_n_iteration, print_loss_avg, jaccard_avg, precision_avg,
                    recall_avg, f1_avg))
            log_file.write(
                'epoch: {}; time: {}; Iteration: {}; train loss: {:.4f}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}\n'.format(
                    epoch, datetime.datetime.now(), trained_n_iteration, print_loss_avg, jaccard_avg, precision_avg,
                    recall_avg, f1_avg))

            if epoch % save_every_epoch == 0:
                torch.save({'epoch': epoch,
                            'iteration': trained_n_iteration,
                            'encoder': encoder.state_dict(),
                            'decoder': decoder.state_dict(),
                            'encoder_optimizer': encoder_optimizer.state_dict(),
                            'decoder_optimizer': decoder_optimizer.state_dict(),
                            }, os.path.join(save_model_path,
                                            'seq2seq_{}_{}_{:.4f}_{:.4f}.checkpoint'.format(epoch, trained_n_iteration,
                                                                                            print_loss_avg, f1_avg)))

        log_file.close()

    def train(self, input_size, hidden_size, encoder_n_layers, encoder_input_embedding_dropout_rate,
              encoder_gru_dropout_rate, encoder_learning_rate, decoder_input_embedding_dropout_rate,
              decoder_learning_rate, teaching_force_rate, save_model_dir, n_epoch=40,
              print_every_iteration=100, save_every_epoch=1, load_model_name=None):
        print('initializing >>>')
        if load_model_name:
            print('load model from checkpoint file:', load_model_name)
            checkpoint = torch.load(load_model_name)

        print('build encoder and decoder >>>')
        encoder = EncoderSeq(self.device, input_size, hidden_size, self.diagnose_count, self.procedure_count,
                             encoder_n_layers, encoder_input_embedding_dropout_rate, encoder_gru_dropout_rate, False)
        decoder = DecoderSeqAttnDe(hidden_size, self.medication_count, decoder_input_embedding_dropout_rate,
                                   self.encoder_hidden_max_length, self.device)

        if load_model_name:
            encoder_sd = checkpoint['encoder']
            decoder_sd = checkpoint['decoder']
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        encoder.train()
        decoder.train()

        print('build optimizer >>>')
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_learning_rate)
        if load_model_name:
            encoder_optimizer_sd = checkpoint['encoder_optimizer']
            decoder_optimizer_sd = checkpoint['decoder_optimizer']
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        patient_records = pd.read_pickle(self.patient_records_file)
        split_point = int(len(patient_records) * params.train_ratio)
        test_count = int(len(patient_records) * params.test_ratio)
        patient_records_train = patient_records[:split_point]
        patient_records_test = patient_records[split_point:split_point + test_count]

        trained_epoch = 0
        trained_iteration = 0
        if load_model_name:
            trained_epoch_sd = checkpoint['epoch']
            trained_iteration_sd = checkpoint['iteration']
            trained_epoch = trained_epoch_sd
            trained_iteration = trained_iteration_sd

        # save_model_structure = str(encoder_n_layers) + '_' + str(input_size) + '_' + str(hidden_size) + '_' + str(
        #     params.PREDICT_PROP) + '_' + str(params.TOPO_PROP) + '_' + decoder_type
        save_model_structure = str(encoder_n_layers) + '_' + str(input_size) + '_' + str(hidden_size)
        save_model_parameters = str(encoder_input_embedding_dropout_rate) + '_' + str(
            encoder_gru_dropout_rate) + '_' + str(encoder_learning_rate) + '_' + str(
            decoder_input_embedding_dropout_rate) + '_' + str(decoder_learning_rate)
        save_model_path = os.path.join(save_model_dir, save_model_structure, save_model_parameters)

        print('start training the model >>>')
        self.trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, teaching_force_rate,
                        patient_records_train, patient_records_test, save_model_path, n_epoch, print_every_iteration,
                        save_every_epoch, trained_epoch, trained_iteration)


def Seq2SetTraining(input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate,
                    encoder_bidirectional, encoder_regular_lambda, encoder_learning_rate, decoder_dropout_rate,
                    decoder_regular_lambda, decoder_learning_rate, hop, attn_type_kv, attn_type_embedding,
                    save_model_dir='data/model/seq2set', proportion_bce=0.9, proportion_multi=0.1, n_epoch=40,
                    print_every_iteration=100, save_every_epoch=1, load_model_name=None):
    model = TrainMedRecSeq2Set(params.device, params.PATIENT_RECORDS_FILE, params.CONCEPTID_FILE,
                               params.EHR_MATRIX_FILE)
    model.train(input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate,
                encoder_bidirectional, encoder_regular_lambda, encoder_learning_rate, decoder_dropout_rate,
                decoder_regular_lambda, decoder_learning_rate, hop, attn_type_kv, attn_type_embedding, save_model_dir,
                proportion_bce, proportion_multi, n_epoch, print_every_iteration, save_every_epoch, load_model_name)


def Seq2SeqTraining(input_size, hidden_size, encoder_n_layers, encoder_input_embedding_dropout_rate,
                    encoder_gru_dropout_rate, encoder_learning_rate, decoder_input_embedding_dropout_rate,
                    decoder_learning_rate, teaching_force_rate, save_model_dir='data/model/seq2seq',
                    n_epoch=100, print_every_iteration=100, save_every_epoch=1, load_model_name=None):
    model = TrainMedRecSeq2Seq(params.device, params.PATIENT_RECORDS_ORDERED_FILE, params.CONCEPTID_FILE,
                               params.SEQUENCE_MAX_LENGTH, params.ENCODER_HIDDEN_MAX_LENGTH)
    model.train(input_size, hidden_size, encoder_n_layers, encoder_input_embedding_dropout_rate,
                encoder_gru_dropout_rate, encoder_learning_rate, decoder_input_embedding_dropout_rate,
                decoder_learning_rate, teaching_force_rate, save_model_dir, n_epoch, print_every_iteration,
                save_every_epoch, load_model_name)


if __name__ == '__main__':
    # Seq2SetTraining(200, 200, 3, 0.4156419, 0.09946605, False, 0, 0.0000589, 0.75956592, 0, 0.00001552, 19, 'general',
    #                 'general', 'data/test', n_epoch=2)
    Seq2SeqTraining(200, 200, 2, 0.33924615, 0.25448997, 0.00000589, 0.24392542, 0.0000196, 0.8,
                    save_model_dir='data/test/seq2seq_predict_teach_0.8_constant', n_epoch=2,
                    load_model_name='data/test/seq2seq_2_348_4.3409_0.2056.checkpoint')
