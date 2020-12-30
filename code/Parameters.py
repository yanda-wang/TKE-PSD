import torch


class Params:
    def __init__(self):
        self.PATIENT_RECORDS_FILE = 'data/records_final.pkl'
        self.PATIENT_RECORDS_ORDERED_FILE = 'data/ordered_patient_records/records_final_ordered_0.9_0.1.pkl'
        self.CONCEPTID_FILE = 'data/voc_final.pkl'
        self.EHR_MATRIX_FILE = 'data/ehr_adj_final.pkl'

        self.PREDICT_PROP = 0.01
        self.TOPO_PROP = 0.01

        self.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.USE_CUDA else "cpu")

        self.MEDICATION_COUNT = 153
        self.DIAGNOSES_COUNT = 1960
        self.PROCEDURES_COUNT = 1432

        self.OPT_SPLIT_TAG_ADMISSION = -1
        self.OPT_SPLIT_TAG_VARIABLE = -2
        self.OPT_MODEL_MAX_EPOCH = 1

        self.train_ratio = 0.01
        self.test_ratio = 0.01

        self.LOSS_PROPORTION_BCE = 0.9
        self.LOSS_PROPORTION_Multi_Margin = 0.1

        self.SEQUENCE_MAX_LENGTH = 24
        self.TEACHING_FORCE_RATE = 0.8
        self.ENCODER_HIDDEN_MAX_LENGTH = 10

        self.SOS_token = 'SOS'
        self.EOS_token = 'EOS'
        self.SOS_id = 0
        self.EOS_id = 1

        self.BEAM_WIDTH = 5
        self.FINAL_BEAM_RESULT_COUNT = 5


def print_params(self):
    print('current parameters:')
