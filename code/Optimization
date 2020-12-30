import skorch
import torch
import random
import sklearn
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch import optim
from skorch.utils import params_for
from warnings import filterwarnings

from Parameters import Params
from Networks import EncoderLinearQuery, DecoderKeyValueGCNMultiEmbedding
from Networks import EncoderSeq, DecoderSeqAttnDe

params = Params()
PATIENT_RECORDS_FILE = params.PATIENT_RECORDS_FILE  # 'data/records_final.pkl
CONCEPTID_FILE = params.CONCEPTID_FILE  # 'data/voc_final'
EHR_MATRIX_FILE = params.EHR_MATRIX_FILE  # 'data/ehr_adj_final.pkl'
DEVICE = params.device  # torch.device("cuda" if USE_CUDA else "cpu")
MEDICATION_COUNT = params.MEDICATION_COUNT  # 147
DIAGNOSES_COUNT = params.DIAGNOSES_COUNT  # 1958
PROCEDURES_COUNT = params.PROCEDURES_COUNT  # 1426

OPT_SPLIT_TAG_ADMISSION = params.OPT_SPLIT_TAG_ADMISSION  # -1
OPT_SPLIT_TAG_VARIABLE = params.OPT_SPLIT_TAG_VARIABLE  # -2
OPT_MODEL_MAX_EPOCH = params.OPT_MODEL_MAX_EPOCH

LOSS_PROPORTION_BCE = params.LOSS_PROPORTION_BCE  # 0.9
LOSS_PROPORTION_MULTI = params.LOSS_PROPORTION_Multi_Margin  # 0.1

SEQ_MAX_LENGTH = params.SEQUENCE_MAX_LENGTH
TEACHING_FORCE_RATE = params.TEACHING_FORCE_RATE

ENCODER_HIDDEN_MAX_LENGTH = params.ENCODER_HIDDEN_MAX_LENGTH

SOS_ID = params.SOS_id
EOS_ID = params.EOS_id


# define the encoder-decoder model
class MedRecSeq2Set(nn.Module):
    def __init__(self, device, input_size, hidden_size, **kwargs):
        super().__init__()
        self.encoder = EncoderLinearQuery(device=device, input_size=input_size, hidden_size=hidden_size,
                                          **params_for('encoder', kwargs))
        self.decoder = DecoderKeyValueGCNMultiEmbedding(device=device, hidden_size=hidden_size,
                                                        **params_for('decoder', kwargs))
        self.device = device

    def split_records(self, x):
        records = []
        split_records = np.split(x, np.where(x == OPT_SPLIT_TAG_ADMISSION)[0])
        admission = split_records[0]
        current_records = []
        split_code = np.split(admission, np.where(admission == OPT_SPLIT_TAG_VARIABLE)[0])
        current_records.append(split_code[0].tolist())
        current_records.append(split_code[1][1:].tolist())
        current_records.append(split_code[2][1:].tolist())
        records.append(current_records)

        for admission in split_records[1:]:
            current_records = []
            split_code = np.split(admission[1:], np.where(admission[1:] == OPT_SPLIT_TAG_VARIABLE)[0])
            current_records.append(split_code[0].tolist())
            current_records.append(split_code[1][1:].tolist())
            current_records.append(split_code[2][1:].tolist())
            records.append(current_records)

        return records

    def forward(self, x):
        records = self.split_records(x)
        patient_representation, keys, values = self.encoder(records)
        predict, _ = self.decoder(patient_representation, keys, values)
        return predict


# warp the encoder-decoder model in skorch
class MedRecTrainer(skorch.NeuralNet):
    def __init__(self, *args, optimizer_encoder=optim.Adam, optimizer_decoder=optim.Adam, **kwargs):
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_decoder = optimizer_decoder
        super().__init__(*args, **kwargs)

    def initialize_optimizer(self, triggered_directly=True):
        kwargs = self._get_params_for('optimizer_encoder')
        self.optimizer_encoder_ = self.optimizer_encoder(self.module_.encoder.parameters(), **kwargs)
        kwargs = self._get_params_for('optimizer_decoder')
        self.optimizer_decoder_ = self.optimizer_decoder(self.module_.decoder.parameters(), **kwargs)

    def train_step(self, Xi, yi, **fit_params):
        yi = skorch.utils.to_numpy(yi).tolist()[0]

        self.module_.train()
        self.optimizer_encoder_.zero_grad()
        self.optimizer_decoder_.zero_grad()

        y_pred = self.infer(Xi)
        loss = self.get_loss(y_pred, yi)
        loss.backward()

        self.optimizer_encoder_.step()
        self.optimizer_decoder_.step()

        return {'loss': loss, 'y_pred': y_pred}

    def infer(self, Xi, yi=None):
        Xi = skorch.utils.to_numpy(Xi)[0]
        return self.module_(Xi)

    def get_loss(self, y_pred, y_true, **kwargs):

        loss_bce_target = np.zeros((1, MEDICATION_COUNT))
        loss_bce_target[:, y_true] = 1
        loss_multi_target = np.full((1, MEDICATION_COUNT), -1)
        for idx, item in enumerate(y_true):
            loss_multi_target[0][idx] = item

        loss_bce = F.binary_cross_entropy_with_logits(y_pred, torch.FloatTensor(loss_bce_target).to(self.device))
        loss_multi = F.multilabel_margin_loss(torch.sigmoid(y_pred),
                                              torch.LongTensor(loss_multi_target).to(self.device))
        loss = LOSS_PROPORTION_BCE * loss_bce + LOSS_PROPORTION_MULTI * loss_multi
        return loss

    def _predict(self, X, most_probable=True):
        filterwarnings('error')
        y_probas = []

        for output in self.forward_iter(X, training=False):
            if most_probable:
                predict_prob = skorch.utils.to_numpy(torch.sigmoid(output))[0]
                predict_multi_hot = predict_prob.copy()

                index_nan = np.argwhere(np.isnan(predict_multi_hot))
                if index_nan.shape[0] != 0:
                    predict_multi_hot = np.zeros_like(predict_multi_hot)

                predict_multi_hot[predict_multi_hot >= 0.5] = 1
                predict_multi_hot[predict_multi_hot < 0.5] = 0
                predict = np.where(predict_multi_hot == 1)[0]
            else:
                predict = skorch.utils.to_numpy(torch.sigmoid(output))[0]
            y_probas.append(predict)

        return np.array(y_probas)

    def predict_proba(self, X):
        return self._predict(X, most_probable=False)

    def predict(self, X):
        return self._predict(X, most_probable=True)


class MedRecSeq2SeqAttnDe(nn.Module):
    def __init__(self, device, input_size, hidden_size, output_size, max_length=SEQ_MAX_LENGTH,
                 teaching_force_rate=TEACHING_FORCE_RATE, encoder_hidden_max_length=ENCODER_HIDDEN_MAX_LENGTH,
                 **kwargs):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder = EncoderSeq(device=device, input_size=input_size, hidden_size=hidden_size,
                                  **params_for('encoder', kwargs))
        self.decoder = DecoderSeqAttnDe(device=device, hidden_size=hidden_size, output_size=output_size,
                                        encoder_hidden_max_length=encoder_hidden_max_length,
                                        **params_for('decoder', kwargs))
        self.max_length = max_length
        self.teaching_force_rate = teaching_force_rate
        self.encoder_hidden_max_length = encoder_hidden_max_length

    def split_records(self, x):
        records = []
        split_records = np.split(x, np.where(x == OPT_SPLIT_TAG_ADMISSION)[0])
        admission = split_records[0]
        current_records = []
        split_code = np.split(admission, np.where(admission == OPT_SPLIT_TAG_VARIABLE)[0])
        current_records.append(split_code[0].tolist())
        current_records.append(split_code[1][1:].tolist())
        current_records.append(split_code[2][1:].tolist())
        records.append(current_records)

        for admission in split_records[1:]:
            current_records = []
            split_code = np.split(admission[1:], np.where(admission[1:] == OPT_SPLIT_TAG_VARIABLE)[0])
            current_records.append(split_code[0].tolist())
            current_records.append(split_code[1][1:].tolist())
            current_records.append(split_code[2][1:].tolist())
            records.append(current_records)

        return records

    def forward(self, x, y=None):
        records = self.split_records(x)
        target_length = self.max_length if y is None else len(y)
        y_predict = []
        y_predict_token = []

        query, encoder_output_tmp = self.encoder(records)
        encoder_output = torch.zeros(self.encoder_hidden_max_length, self.hidden_size, device=self.device)

        if encoder_output_tmp.size(0) <= self.encoder_hidden_max_length:
            encoder_output[:encoder_output_tmp.size(0)] = encoder_output_tmp
        else:
            encoder_output = encoder_output_tmp[-self.encoder_hidden_max_length:]

        decoder_input = torch.LongTensor([[SOS_ID]]).to(self.device)
        decoder_hidden = query.unsqueeze(dim=1)  # dim=(1,1,hidden)

        use_teaching_force = y is not None and random.random() < self.teaching_force_rate
        if use_teaching_force:
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
                y_predict.append(decoder_output)
                decoder_input = torch.LongTensor([[y[di]]]).to(self.device)
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
                y_predict.append(decoder_output)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                if decoder_input.item() == EOS_ID:
                    break
        result = torch.stack(y_predict, dim=1)  # dim=(1,#medication,output_size)
        return result


class MedRecSeq2SeqTrainer(skorch.NeuralNet):
    def __init__(self, *args, optimizer_encoder=torch.optim.Adam, optimizer_decoder=torch.optim.Adam, **kwargs):
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_decoder = optimizer_decoder
        super().__init__(*args, **kwargs)

    def initialize_optimizer(self, triggered_directly=True):
        kwargs = self._get_params_for('optimizer_encoder')
        self.optimizer_encoder_ = self.optimizer_encoder(self.module_.encoder.parameters(), **kwargs)
        kwargs = self._get_params_for('optimizer_decoder')
        self.optimizer_decoder_ = self.optimizer_decoder(self.module_.decoder.parameters(), **kwargs)

    def train_step(self, Xi, yi, **fit_params):
        self.module_.train()
        self.optimizer_encoder_.zero_grad()
        self.optimizer_decoder_.zero_grad()
        y_pred = self.infer(Xi, yi)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        loss.backward()
        self.optimizer_encoder_.step()
        self.optimizer_decoder_.step()

        return {'loss': loss, 'y_pred': y_pred}

    def infer(self, Xi, yi=None):
        Xi = skorch.utils.to_numpy(Xi)[0]
        yi = skorch.utils.to_numpy(yi)[0] if yi is not None else None
        return self.module_(Xi, yi)

    def get_loss(self, y_pred, y_true, **kwargs):
        y_true = y_true.long()
        y_true = y_true[:, :y_pred.size(1)]
        y_pred_flat = y_pred.view(y_pred.size(0) * y_pred.size(1),
                                  -1)  # change dim(y_pred) from (1,#medication,MED_COUNT) to (#medication,MED_COUNT)
        y_true_flat = y_true.view(
            y_true.size(0) * y_true.size(1))  # change dim(y_true) from (1,#medication) to (medication)

        y_pred_flat = skorch.utils.to_tensor(y_pred_flat, device=self.device)
        y_true_flat = skorch.utils.to_tensor(y_true_flat, device=self.device)

        return super().get_loss(y_pred_flat, y_true_flat, **kwargs)

    def _predict(self, X, most_probable=True):
        y_probs = []
        for y_predict in self.forward_iter(X, training=False):
            if most_probable:
                pad = np.zeros((y_predict.size(0), SEQ_MAX_LENGTH))
                pad[:, :y_predict.size(1)] = skorch.utils.to_numpy(y_predict.max(-1)[-1])
            else:
                pad = np.zeros((y_predict.size(0), SEQ_MAX_LENGTH, y_predict.size(-1)))
                pad[:, :y_predict.size(1)] = skorch.utils.to_numpy(y_predict)
            y_probs.append(pad)
        y_prob = np.concatenate(y_probs, 0)
        return y_prob

    def predict_proba(self, X):
        return self._predict(X, most_probable=False)

    def predict(self, X):
        return self._predict(X, most_probable=True)

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = y_pred.copy()

        for i, yi in enumerate(y):
            yi = skorch.utils.to_numpy(yi.squeeze())
            y_true[:, :len(yi)] = yi

        return sklearn.metrics.accuracy_score(y_true.flatten(), y_pred.flatten())
