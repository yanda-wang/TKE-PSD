# we borrow the code from https://github.com/sjy1203/GAMENet

import dill
import pandas as pd
import numpy as np

med_file = 'data/PRESCRIPTIONS.csv'
diag_file = 'data/DIAGNOSES_ICD.csv'
procedure_file = 'data/PROCEDURES_ICD.csv'

ndc2atc_file = 'data/ndc2atc_level4.csv'
cid_atc = 'data/drug-atc.csv'
ndc2rxnorm_file = 'data/ndc2rxnorm_mapping.txt'

voc_file = 'data/voc_final.pkl'
data_path = 'data/records_final.pkl'

# drug-drug interactions can be down https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?dl=0
ddi_file = 'data/drug-DDI.csv'


def process_procedure():
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE': 'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)
    #     pro_pd = pro_pd[pro_pd['SEQ_NUM']<5]
    #     def icd9_tree(x):
    #         if x[0]=='E':
    #             return x[:4]
    #         return x[:3]
    #     pro_pd['ICD9_CODE'] = pro_pd['ICD9_CODE'].map(icd9_tree)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


def process_med():
    med_pd = pd.read_csv(med_file, dtype={'NDC': 'category'})
    # filter
    med_pd.drop(columns=['ROW_ID', 'DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
                         'FORMULARY_DRUG_CD', 'GSN', 'PROD_STRENGTH', 'DOSE_VAL_RX',
                         'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'FORM_UNIT_DISP',
                         'ROUTE', 'ENDDATE', 'DRUG'], axis=1, inplace=True)
    med_pd.drop(index=med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    def filter_first24hour_med(med_pd):
        med_pd_new = med_pd.drop(columns=['NDC'])
        med_pd_new = med_pd_new.groupby(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']).head([1]).reset_index(drop=True)
        med_pd_new = pd.merge(med_pd_new, med_pd, on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'])
        med_pd_new = med_pd_new.drop(columns=['STARTDATE'])
        return med_pd_new

    med_pd = filter_first24hour_med(med_pd)
    #     med_pd = med_pd.drop(columns=['STARTDATE'])

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    # visit > 2
    def process_visit_lg2(med_pd):
        a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
        a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x: len(x))
        a = a[a['HADM_ID_Len'] > 1]
        return a

    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')

    return med_pd.reset_index(drop=True)


def process_diag():
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM', 'ROW_ID'], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    return diag_pd.reset_index(drop=True)


def ndc2atc4(med_pd):
    with open(ndc2rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index=med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)

    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4': 'NDC'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(pro_count.loc[:1000, 'ICD9_CODE'])]

    return pro_pd.reset_index(drop=True)


def filter_2000_most_diag(diag_pd):
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]

    return diag_pd.reset_index(drop=True)


def filter_300_most_med(med_pd):
    med_count = med_pd.groupby(by=['NDC']).size().reset_index().rename(columns={0: 'count'}).sort_values(by=['count'],
                                                                                                         ascending=False).reset_index(
        drop=True)
    med_pd = med_pd[med_pd['NDC'].isin(med_count.loc[:299, 'NDC'])]

    return med_pd.reset_index(drop=True)


def process_all():
    # get med and diag (visit>=2)
    med_pd = process_med()
    med_pd = ndc2atc4(med_pd)
    #     med_pd = filter_300_most_med(med_pd)

    diag_pd = process_diag()
    diag_pd = filter_2000_most_diag(diag_pd)

    pro_pd = process_procedure()
    #     pro_pd = filter_1000_most_pro(pro_pd)

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index()
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(
        columns={'ICD9_CODE': 'PRO_CODE'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    #     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x))
    return data


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()

    med_voc.add_sentence(['SOS', 'EOS'])
    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE'])
        med_voc.add_sentence(row['NDC'])
        pro_voc.add_sentence(row['PRO_CODE'])

    dill.dump(obj={'diag_voc': diag_voc, 'med_voc': med_voc, 'pro_voc': pro_voc}, file=open('voc_final.pkl', 'wb'))
    return diag_voc, med_voc, pro_voc


def create_patient_record(df, diag_voc, med_voc, pro_voc):
    records = []  # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['ICD9_CODE']])
            admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['NDC']])
            patient.append(admission)
        records.append(patient)
    dill.dump(obj=records, file=open('records_final.pkl', 'wb'))
    return records


def create_ehr_adj():
    med_voc = dill.load(open('data/voc_final.pkl', 'rb'))['med_voc']
    med_voc_size = len(med_voc.idx2word)
    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    records = dill.load(open('data/records_final.pkl', 'rb'))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j <= i:
                        continue
                    ehr_adj[med_i, med_j] = 1
                    ehr_adj[med_j, med_i] = 1
    dill.dump(ehr_adj, open('data/ehr_adj_final.pkl', 'wb'))


if __name__ == '__main__':
    # data = process_all()
    # # statistics()
    # data.to_pickle('data/data_final.pkl')
    # data.head()

    # path = 'data/data_final.pkl'
    # df = pd.read_pickle(path)
    # diag_voc, med_voc, pro_voc = create_str_token_mapping(df)
    # records = create_patient_record(df, diag_voc, med_voc, pro_voc)
    # print(len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    # create_ehr_adj()

    #################################################################
    # patient_records = dill.load(open('data/records_final.pkl', 'rb'))
    # for patient in patient_records[:5]:
    #     print(patient)
    #
    # voc = dill.load(open('data/voc_final.pkl', 'rb'))
    # med_voc = voc['med_voc']
    # print(med_voc.word2idx)
    # print(len(med_voc.word2idx))

    ehr_graph = dill.load(open('data/ehr_adj_final.pkl', 'rb'))
    print(ehr_graph)
