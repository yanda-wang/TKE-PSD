import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.parameter import Parameter


class EncoderLinearQuery(nn.Module):
    def __init__(self, device, input_size, hidden_size, diagnoses_count, procedures_count, n_layers=1,
                 embedding_dropout_rate=0, gru_dropout_rate=0, bidirectional=False):
        super(EncoderLinearQuery, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_diagnoses = nn.Embedding(diagnoses_count, input_size)
        self.embedding_procedures = nn.Embedding(procedures_count, input_size)
        self.n_layers = n_layers
        self.embedding_dropout_rate = embedding_dropout_rate
        self.gru_dropout_rate = gru_dropout_rate
        self.bidirectional = bidirectional
        self.gru_diagnoses = nn.GRU(self.input_size, self.hidden_size, self.n_layers,
                                    dropout=(0 if self.n_layers == 1 else self.gru_dropout_rate),
                                    bidirectional=self.bidirectional)
        self.gru_procedures = nn.GRU(self.input_size, self.hidden_size, self.n_layers,
                                     dropout=(0 if self.n_layers == 1 else self.gru_dropout_rate),
                                     bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(self.embedding_dropout_rate)
        if bidirectional:
            self.linear_embedding = nn.Sequential(nn.ReLU(), nn.Linear(2 * 2 * hidden_size, 2 * hidden_size), nn.ReLU(),
                                                  nn.Linear(2 * hidden_size, hidden_size))
        else:
            self.linear_embedding = nn.Sequential(nn.ReLU(), nn.Linear(2 * hidden_size, hidden_size))

    def forward(self, patient_record):

        seq_diagnoses = []
        seq_procedures = []
        memory_values = []
        for admission in patient_record:
            data_diagnoses = self.dropout(
                self.embedding_diagnoses(torch.LongTensor(admission[0]).to(self.device))).mean(dim=0, keepdim=True)
            data_procedures = self.dropout(
                self.embedding_diagnoses(torch.LongTensor(admission[1]).to(self.device))).mean(dim=0, keepdim=True)
            seq_diagnoses.append(data_diagnoses)
            seq_procedures.append(data_procedures)
            memory_values.append(admission[2])
        seq_diagnoses = torch.cat(seq_diagnoses).unsqueeze(dim=1)  # dim=(#admission,1,input_size)
        seq_procedures = torch.cat(seq_procedures).unsqueeze(dim=1)  # dim=(#admission,1,input_size)

        # output dim=(#admission,1,num_direction*hidden_size)
        # hidden dim=(num_layers*num_directions,1,hidden_size)
        output_diagnoses, hidden_diagnoses = self.gru_diagnoses(seq_diagnoses)
        output_procedures, hidden_procedures = self.gru_procedures(seq_procedures)
        patient_representations = torch.cat((output_diagnoses, output_procedures), dim=-1).squeeze(
            dim=1)  # dim=(#admission,2*hidden_size*num_direction)

        queries = self.linear_embedding(patient_representations)  # dim=(#admission,hidden_size)
        query = queries[-1:]  # linear representation of the last admission, dim=(1,hidden_size)

        if len(patient_record) > 1:  # more than one admission
            memory_keys = queries[:-1]  # dim=(#admission-1,hidden_size)
            memory_values = memory_values[:-1]  # a list of list, medications except for the last admission
        else:
            memory_keys = None
            memory_values = None

        return query, memory_keys, memory_values


class EncoderSeq(nn.Module):
    def __init__(self, device, input_size, hidden_size, diagnoses_count, procedures_count, n_layers=1,
                 embedding_dropout_rate=0, gru_dropout_rate=0, bidirectional=False):
        super(EncoderSeq, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_diagnoses = nn.Embedding(diagnoses_count, input_size)
        self.embedding_procedures = nn.Embedding(procedures_count, input_size)
        self.n_layers = n_layers
        self.embedding_dropout_rate = embedding_dropout_rate
        self.gru_dropout_rate = gru_dropout_rate
        self.bidirectional = bidirectional
        self.gru_diagnoses = nn.GRU(self.input_size, self.hidden_size, self.n_layers,
                                    dropout=(0 if self.n_layers == 1 else self.gru_dropout_rate),
                                    bidirectional=self.bidirectional)
        self.gru_procedures = nn.GRU(self.input_size, self.hidden_size, self.n_layers,
                                     dropout=(0 if self.n_layers == 1 else self.gru_dropout_rate),
                                     bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(self.embedding_dropout_rate)
        if bidirectional:
            self.linear_embedding = nn.Sequential(nn.ReLU(), nn.Linear(2 * 2 * hidden_size, 2 * hidden_size), nn.ReLU(),
                                                  nn.Linear(2 * hidden_size, hidden_size))
        else:
            self.linear_embedding = nn.Sequential(nn.ReLU(), nn.Linear(2 * hidden_size, hidden_size))

    def forward(self, patient_record):

        seq_diagnoses = []
        seq_procedures = []
        memory_values = []
        for admission in patient_record:
            data_diagnoses = self.dropout(
                self.embedding_diagnoses(torch.LongTensor(admission[0]).to(self.device))).mean(dim=0, keepdim=True)
            data_procedures = self.dropout(
                self.embedding_diagnoses(torch.LongTensor(admission[1]).to(self.device))).mean(dim=0, keepdim=True)
            seq_diagnoses.append(data_diagnoses)
            seq_procedures.append(data_procedures)
            memory_values.append(admission[2])
        seq_diagnoses = torch.cat(seq_diagnoses).unsqueeze(dim=1)  # dim=(#admission,1,input_size)
        seq_procedures = torch.cat(seq_procedures).unsqueeze(dim=1)  # dim=(#admission,1,input_size)

        # output dim=(#admission,1,num_direction*hidden_size)
        # hidden dim=(num_layers*num_directions,1,hidden_size)
        output_diagnoses, hidden_diagnoses = self.gru_diagnoses(seq_diagnoses)
        output_procedures, hidden_procedures = self.gru_procedures(seq_procedures)
        patient_representations = torch.cat((output_diagnoses, output_procedures), dim=-1).squeeze(
            dim=1)  # dim=(#admission,2*hidden_size*num_direction)

        queries = self.linear_embedding(patient_representations)  # dim=(#admission,hidden_size)
        query = queries[-1:]  # linear representation of the last admission, dim=(1,hidden_size)

        return query, queries


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method,
                             "is not an appropriate attention method, choose from dot, general, and concat.")

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    # score=query.T()*keys
    def dot_score(self, query, keys):
        return torch.sum(query * keys, -1).unsqueeze(0)  # dim=(1,keys.dim(0))

    # score=query.T()*W*keys, W is a matrix
    def general_score(self, query, keys):
        energy = self.attn(keys)
        return torch.sum(query * energy, -1).unsqueeze(0)  # dim=(1, keys.dim(0))

    # score=v.T()*tanh(W*[query;keys])
    def concat_score(self, query, keys):
        energy = self.attn(torch.cat((query.expand(keys.size(0), -1), keys), -1)).tanh()
        return torch.sum(self.v * energy, -1).unsqueeze(0)  # dim=(1, keys.dim(0)

    def initialize_weights(self, init_range):
        if self.method == 'concat':
            self.v.data.uniform_(-init_range, init_range)

    def forward(self, query, keys):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(query, keys)
        elif self.method == 'concat':
            attn_energies = self.concat_score(query, keys)
        elif self.method == 'dot':
            attn_energies = self.dot_score(query, keys)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1)  # dim=(1,keys.dim(0))


class DecoderKeyValueGCNMultiEmbedding(nn.Module):
    def __init__(self, device, hidden_size, output_size, medication_count, hop=1, dropout_rate=0, attn_type_kv='dot',
                 attn_type_embedding='dot', ehr_adj=None):
        super(DecoderKeyValueGCNMultiEmbedding, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.medication_count = medication_count
        self.hop_count = hop
        self.dropout_rate = dropout_rate
        self.attn_type_kv = attn_type_kv
        self.attn_type_embedding = attn_type_embedding
        self.ehr_adj = ehr_adj

        self.ehr_gcn = GCN(self.device, self.medication_count, self.hidden_size, self.ehr_adj, self.dropout_rate)
        self.attn_kv = Attn(self.attn_type_kv, hidden_size)
        self.attn_embedding = Attn(self.attn_type_embedding, hidden_size)
        self.output = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size * 3, hidden_size * 2), nn.ReLU(),
                                    nn.Linear(hidden_size * 2, output_size))

    def forward(self, query, memory_keys, memory_values):
        if memory_keys is None:
            embedding_medications = self.ehr_gcn()
            weights_embedding = self.attn_embedding(query, embedding_medications)
            context_e = torch.mm(weights_embedding, embedding_medications)
            context_o = context_e
        else:
            memory_values_multi_hot = np.zeros((len(memory_values), self.medication_count))
            for idx, admission in enumerate(memory_values):
                memory_values_multi_hot[idx, admission] = 1
            memory_values_multi_hot = torch.FloatTensor(memory_values_multi_hot).to(self.device)

            embedding_medications = self.ehr_gcn()
            weights_kv = self.attn_kv(query, memory_keys)
            weighted_values = weights_kv.mm(memory_values_multi_hot)
            current_o = torch.mm(weighted_values, embedding_medications)
            context_o = torch.add(query, current_o)
            for hop in range(1, self.hop_count):
                embedding_medications = self.ehr_gcn()
                weights_kv = self.attn_kv(context_o, memory_keys)
                weighted_values = weights_kv.mm(memory_values_multi_hot)
                current_o = torch.mm(weighted_values, embedding_medications)
                context_o = torch.add(context_o, current_o)
            embedding_medications = self.ehr_gcn()
            weights_embedding = self.attn_embedding(query, embedding_medications)
            context_e = torch.mm(weights_embedding, embedding_medications)
        output = self.output(torch.cat([query, context_o, context_e], -1))
        predict_prob = torch.sigmoid(output).detach().cpu().numpy()[0]

        return output, predict_prob


class DecoderSeqAttnDe(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_rate_input_med, encoder_hidden_max_length,
                 device=torch.device('cpu:0')):
        super(DecoderSeqAttnDe, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate_input_med
        self.encoder_hidden_max_length = encoder_hidden_max_length
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.encoder_hidden_max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_medication, last_hidden, encoder_output):
        embedded = self.embedding(input_medication).to(self.device).view(1, 1, -1)  # dim=(1,1,hidden_size)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], last_hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_output.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)  # dim=(1,1,hidden_size)
        output = F.relu(output)
        output, hidden = self.gru(output, last_hidden)

        output = self.out(output[0])  # dim=(1,output_size)
        output_prob = F.softmax(output, dim=-1).detach().cpu().numpy()
        # output = F.log_softmax(self.out(output[0]), dim=1)  # dim=(1,output_size)
        output = F.log_softmax(output, dim=1)  # dim=(1,output_size)
        return output, hidden, output_prob


"""
fundamental components for GCN
"""


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, device, item_count, embedding_size, adj_matrix, dropout_rate):
        super(GCN, self).__init__()
        self.device = device
        self.item_count = item_count
        self.embedding_size = embedding_size

        adj_matrix = self.normalize(adj_matrix + np.eye(adj_matrix.shape[0]))
        self.adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)
        self.x = torch.eye(item_count).to(self.device)

        self.gcn1 = GraphConvolution(item_count, embedding_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.gcn2 = GraphConvolution(embedding_size, embedding_size)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj_matrix)  # dim=(item_count,embedding*size)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj_matrix)  # dim=(item_count,embedding_size)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


if __name__ == '__main__':
    embedding = nn.Embedding(2, 3)
