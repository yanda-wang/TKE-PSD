import mathimport torchimport dillimport osimport randomimport numpy as npimport pandas as pdfrom tqdm import tqdmfrom Networks import EncoderLinearQuery, DecoderKeyValueGCNMultiEmbeddingfrom Parameters import Paramsparams = Params()EHR_MATRIX_FILE = params.EHR_MATRIX_FILEDIAG_COUNT = params.DIAGNOSES_COUNTPRO_COUNT = params.PROCEDURES_COUNTMED_COUNT = params.MEDICATION_COUNTDEVICE = params.deviceclass Orders:    def __init__(self, potential_sigma=0.4721, device=DEVICE):        self.sigma = potential_sigma        self.device = device    def get_shortest_distance(self, EHR_matrix):        node_count = np.size(EHR_matrix, 0)        distance = np.zeros_like(EHR_matrix)        for i in range(node_count):            for j in range(node_count):                if i == j:                    distance[i][j] = 0                else:                    if EHR_matrix[i][j] == 0:                        distance[i][j] = math.inf                    else:                        distance[i][j] = 1        for i in range(node_count):            for j in range(node_count):                for k in range(j + 1, node_count):                    if distance[j][k] > distance[j][i] + distance[i][k]:                        distance[j][k] = distance[j][i] + distance[i][k]                        distance[k][j] = distance[j][i] + distance[i][k]        return distance    def get_topological_potential(self, EHR_matrix):        shortest_distance = self.get_shortest_distance(EHR_matrix)        node_count = np.size(EHR_matrix, 0)        potential = []        for i in range(node_count):            current_potential = 0            for j in range(node_count):                current_potential += math.exp(-shortest_distance[i][j] / self.sigma)            potential.append(current_potential)        potential = np.array(potential)        max_index, min_index = np.argmax(potential), np.argmin(potential)        max_value, min_value = potential[max_index], potential[min_index]        potential = [(p - min_value) / (max_value - min_value) for p in potential]        return potential    def get_ordered_patient_records(self, diag_count, pro_count, med_count, input_size, hidden_size, encoder_n_layers,                                    encoder_embedding_dropout_rate, encoder_gru_dropout_rate, decoder_attn_type_kv,                                    decoder_attn_type_embedding, decoder_dropout_rate, decoder_hop_count,                                    patient_records_file, save_ordered_records_file, load_model_file, ehr_adj_file,                                    predict_prop=0.9, topo_prop=0.1, model_structure=None, model_parameters=None):        print('initializing models >>>')        ehr_adj = np.load(ehr_adj_file)        encoder = EncoderLinearQuery(self.device, input_size, hidden_size, diag_count, pro_count, encoder_n_layers,                                     encoder_embedding_dropout_rate, encoder_gru_dropout_rate)        decoder = DecoderKeyValueGCNMultiEmbedding(self.device, hidden_size, med_count, med_count, decoder_hop_count,                                                   decoder_dropout_rate, decoder_attn_type_kv,                                                   decoder_attn_type_embedding, ehr_adj)        checkpoint = torch.load(load_model_file)        encoder_sd = checkpoint['encoder']        decoder_sd = checkpoint['decoder']        encoder.load_state_dict(encoder_sd)        decoder.load_state_dict(decoder_sd)        encoder.to(self.device)        decoder.to(self.device)        encoder.eval()        decoder.eval()        topological_potential = self.get_topological_potential(ehr_adj)        patient_records = pd.read_pickle(patient_records_file)        ordered_patient_records = []        print('calculating potential >>>')        for i, patient in enumerate(tqdm(patient_records)):            current_ordered_patient = []            for idx, item in enumerate(patient):                current_records = patient[:idx + 1]                target_medication = current_records[-1][2]                query, memory_keys, memory_values = encoder(current_records)                predict_output, predict_potential = decoder(query, memory_keys, memory_values)                selected_predict_potential = [predict_potential[med_idx] for med_idx in target_medication]                ######################                max_index, min_index = np.argmax(selected_predict_potential), np.argmin(selected_predict_potential)                max_value, min_value = selected_predict_potential[int(max_index)], selected_predict_potential[                    int(min_index)]                if len(target_medication) != 1:                    selected_predict_potential = [(attn - min_value) / (max_value - min_value) for attn in                                                  selected_predict_potential]                ##############################################                selected_topological_potential = [topological_potential[med_index] for med_index in target_medication]                final_potential = [predict_prop * attn + topo_prop * topo for (attn, topo) in                                   zip(selected_predict_potential, selected_topological_potential)]                ordered_medications_final_potential = {}                ordered_medications_predict_potential = {}                ordered_medications_topo_potential = {}                for med, final, attn, topo in zip(target_medication, final_potential, selected_predict_potential,                                                  selected_topological_potential):                    ordered_medications_final_potential[med] = final                    ordered_medications_predict_potential[med] = attn                    ordered_medications_topo_potential[med] = topo                ordered_medications_final_potential = sorted(ordered_medications_final_potential.items(),                                                             key=lambda p_item: p_item[1], reverse=True)                current_ordered_medications = []                current_ordered_final_potential = []                current_ordered_predict_potential = []                current_ordered_topo_potential = []                for (key, value) in ordered_medications_final_potential:                    current_ordered_medications.append(key)                    current_ordered_final_potential.append(value)                    current_ordered_predict_potential.append(ordered_medications_predict_potential[key])                    current_ordered_topo_potential.append(ordered_medications_topo_potential[key])                admission = current_records[-1]                admission.append(current_ordered_medications)                admission.append(current_ordered_final_potential)                admission.append(current_ordered_predict_potential)                admission.append(current_ordered_topo_potential)                current_ordered_patient.append(admission)            ordered_patient_records.append(current_ordered_patient)        if model_structure is not None:            save_ordered_records_file = os.path.join(save_ordered_records_file, model_structure, model_parameters)        if not os.path.exists(save_ordered_records_file):            os.makedirs(save_ordered_records_file)        print('saving ordered patient records to:', save_ordered_records_file)        dill.dump(obj=ordered_patient_records,                  file=open(save_ordered_records_file + '/records_final_ordered_' + str(predict_prop) + '_' + str(                      topo_prop) + '.pkl', 'wb'))    def order_randomly(self, patient_records_file, save_ordered_patient_records_file):        patient_records = pd.read_pickle(patient_records_file)        new_patient_records = []        for i, patient in enumerate(tqdm(patient_records)):            current_ordered_patient = []            for adm in patient:                medications = [item for item in adm[2]]                random.shuffle(medications)                new_adm = [item for item in adm]                new_adm.append(medications)                current_ordered_patient.append(new_adm)            new_patient_records.append(current_ordered_patient)        if not os.path.exists(save_ordered_patient_records_file):            os.makedirs(save_ordered_patient_records_file)        dill.dump(obj=new_patient_records,                  file=open(save_ordered_patient_records_file + '/records_final_ordered_randomly.pkl', 'wb'))    def get_med_frequency(self, patient_records):        med_frequency = {}        for patient in patient_records:            for adm in patient:                medications = adm[2]                for med in medications:                    if med in med_frequency.keys():                        med_frequency[med] += 1                    else:                        med_frequency[med] = 1        return med_frequency    def order_by_frequency(self, patient_records_file, save_ordered_patient_records_file, frequency_first=True):        patient_records = pd.read_pickle(patient_records_file)        med_frequency = self.get_med_frequency(patient_records)        ordered_patient = []        for patient in patient_records:            current_patient = []            for adm in patient:                medications = {}                for med in adm[2]:                    medications[med] = med_frequency[med]                medications = sorted(medications.items(), key=lambda p_item: p_item[1], reverse=frequency_first)                ordered_medications = []                for (key, value) in medications:                    ordered_medications.append(key)                adm.append(ordered_medications)                current_patient.append(adm)            ordered_patient.append(current_patient)        if not os.path.exists(save_ordered_patient_records_file):            os.makedirs(save_ordered_patient_records_file)        if frequency_first:            file_type = 'frequency'        else:            file_type = 'rare'        dill.dump(obj=ordered_patient,                  file=open(save_ordered_patient_records_file + '/records_final_ordered_' + file_type + '.pkl', 'wb'))    def order_rare_first(self, patient_records_file, save_ordered_patient_file):        self.order_by_frequency(patient_records_file, save_ordered_patient_file, frequency_first=False)    def order_frequency_first(self, patient_records_file, save_ordered_patient_file):        self.order_by_frequency(patient_records_file, save_ordered_patient_file, frequency_first=True)if __name__ == '__main__':    module = Orders()    module.get_ordered_patient_records(DIAG_COUNT, PRO_COUNT, MED_COUNT, 200, 200, 3, 0.001, 0.001, 'general',                                       'general', 0.01, 4, params.PATIENT_RECORDS_FILE, 'data/test',                                       'data/test/medrec_8_98584_0.8003_0.8484_0.6384.checkpoint',                                       params.EHR_MATRIX_FILE)    # module.order_predict_potential(DIAG_COUNT, PRO_COUNT, MED_COUNT, 200, 200, 3, 0.09946605, 0.4156419, 'general',    #                                'general', 0.75956592, 19, params.PATIENT_RECORDS_FILE,    #                                'data/model/medrec_40_492920_0.6940_0.7952_0.6660.checkpoint',    #                                'data/ordered_patient_records', params.EHR_MATRIX_FILE,    #                                '3_200_200_False_general_general',    #                                '0.4156419_0.09946605_0_5.89e-05_0.75956592_0_1.552e-05_19')