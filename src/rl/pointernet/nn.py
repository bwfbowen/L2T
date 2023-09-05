import math 
import numpy as np
import torch 
import torch.nn as nn
from torch.autograd import Variable


# mask already search index
def mask_logit(logit, mask, prev_idx):
    '''
    mask logit probability if they are in previous indice
    Args:
    logit (tensor): from attention output with shape (batch_size, sequence length)
    mask (tensor or None): selected mask
    prev_idx (tensor): None or previous retrieved indice (batch_size, )

    Return:
    logit (tensor): same shape with input, but mask elements

    '''
    req = 4
    batch_size = logit.size(0)
    zeros = torch.zeros(batch_size)
    mask_copy = mask.clone()
    mask_copy_0 = mask.clone()

    # first index should be depot
    logit = logit.clone()

    # if prev_idx is None:
    #     mask_copy[[b for b in range(batch_size)], 1:] = 1
    #     print(logit)
    #     mask_copy_0[[b for b in range(batch_size)], 1:] = 0
    #     logit[mask_copy_0] = 5
    #     print(mask_copy[34])
    #     return logit, mask_copy

    # batch_size = logit.size(0)
    if prev_idx is not None:
        # print(prev_idx)
        mask_copy[[b for b in range(batch_size)], prev_idx.data] = 1
        # print("mask:",mask_copy[34])
        logit[mask_copy] = -np.inf
        # print("logit:",logit[34])

    return logit, mask_copy


def greedy(probs, embedding_x):
    '''
    greedy search and return the index with
    the biggest probability used in validation set

    Args:
    probs (tensor): (batch_size, k)
    embedding_x (tensor): (k, batch_size, embedding_dim)

    Return:
    -- new_decoder_input (tensor): selected embedding x with shape (batch_size, embedding_dim)
    as new decoder_input
    -- idx (tensor): selected idx tensor (k,)
    '''
    batch_size = probs.size(0)
    idx = torch.argmax(probs, dim = 1).long()
    new_decoder_input = embedding_x[[x for x in range(batch_size)], idx.data, :]
    return idx, new_decoder_input


def sampling(probs, embedding_x, prev_idx):
    '''
    sampling indice from probability used in train and validation is ok

    Args:
    probs (tensor): (batch_size, k)
    embedding_x (tensor): (k, batch_size, embedding_dim)
    prev_idx (list): list of previous index (batch_size, 1), should be LongTensor

    Return:
    -- new_decoder_input (tensor): selected embedding x with shape (batch_size, embedding_dim)
    as new decoder_input
    -- idx (tensor): selected idx tensor (k,)
    '''
    batch_size = probs.size(0)
    idx = probs.multinomial(1).squeeze(1).long()

    def is_exist(idx, prev_idx):
        for old_idx in prev_idx:
            if old_idx.eq(idx).data.any():
                return True
        return False

    while is_exist(idx, prev_idx):
        idx = probs.multinomial(1).squeeze(1).long()

    return idx, embedding_x[[x for x in range(batch_size)], idx.data, :]


def softmax_temperature(logit_mask, T = 2.):
    '''
    Implement softmax temperature strategy

    Args:
    logit_mask (batch_size, seq_len)
    T (float): temperature
    '''
    return logit_mask / T


# define tour length function as reward
def tour_length(pi, use_cuda = False):
    '''
    calculate the total length of the tour
    Args:
    pi (list): length is sequence length,
                the element shape is (batch_size, point_size)
    Return:
    tour_len (tensor): (batch_size, 1)
    '''

    n = len(pi)
    batch_size = pi[0].size(0)
    tour_len = Variable(torch.zeros(batch_size))

    if use_cuda:
        tour_len = tour_len.cuda()

    for i in range(n-1):
        tour_len += torch.norm(pi[i+1] - pi[i], p = 2, dim = 1)
    tour_len += torch.norm(pi[n-1] - pi[0], p = 2, dim = 1)
    return tour_len


def infeasible_punishment(index_list, precedence, punishment: float = 1e5):
    _punishment = 0.
    seq_id_dict = {int(index): idx for idx, index in enumerate(index_list)}
    is_feasible = True if seq_id_dict[0] == 0 and seq_id_dict[1] == 1 else False 
    for o, d in precedence.items():
        if seq_id_dict[o] > seq_id_dict[d]:
            is_feasible = False 
            break 
    if not is_feasible:
        _punishment = punishment 
    return _punishment    


class Encoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        # use default number of layers is 1
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, embedding_x):
        '''
        Args:
        embedding_x (tensor): shape (sequence length, batch, input_dim)
        hidden (tensor): shape (num_layer * num_direction, batch_size, hidden_dim)

        Return:
        output (tensor): (sequence length, batch_size, hidden_dim)
        next_hidden (tensor): (num_layer * num_direction, batch_size, hidden_dim)
        '''
        output, next_hidden = self.lstm(embedding_x)
        return output, next_hidden
    

class Attention(nn.Module):
    def __init__(self, dim, C = 10, use_logit_clip = True, use_cuda = False):
        super().__init__()
        self.w_q = nn.Linear(dim, dim) # for query (batch_size, d)
        self.w_ref = nn.Conv1d(dim, dim, 1, 1) # for reference (batch_size, d, k)
        V = torch.Tensor(dim).float() # to trainable parameters
        if use_cuda:
            V = V.cuda()
        # initialize v by uniform in almost (-0.08, 0.08) from the paper
        self.v = nn.Parameter(V)
        self.v.data.uniform_(- 1. / math.sqrt(dim), 1. / math.sqrt(dim))
        self.tanh = nn.Tanh()
        self.C = C
        self.dim = dim
        self.use_logit_clip = use_logit_clip

    def forward(self, encoder_output, query):
        '''
        Args:
        k is sequence length
        encoder_output (tensor): shape is (k, batch_size, dim) from encoder_output
        query (tensor): shape is (batch_size, dim)

        Return:
        ref (tensor): shape is (batch_size, dim, k)
        logit (tensor): probability with shape (batch_size, k)
        '''
        batch_size = query.size(0)
        encoder_output = encoder_output.permute(1, 2, 0)
        # make sure ref shape is (batch_size, dim, k)

        q = self.w_q(query).unsqueeze(2)
        ref = self.w_ref(encoder_output) # batch_size, hidden_dim, k
        k = ref.size(2)
        expanded_q = q.repeat(1, 1, k)
        expanded_v = self.v.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        # batch matrix multiply (batch_size, sequence_length)
        u = torch.bmm(expanded_v, self.tanh(ref + expanded_q)).squeeze(1)

        if self.use_logit_clip:
            logit = self.C * self.tanh(u)
        else:
            logit = u

        # return ref for glimpse
        return logit    
    

class Decoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, sequence_length,
                 decoder_type = "sampling", num_glimpse = 1, use_cuda = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_glimpse = num_glimpse
        self.sequence_length = sequence_length

        # lstm cell weights
        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_dim)

        # pointer and glimpse
        self.pointer = Attention(hidden_dim, use_cuda = use_cuda)
        self.glimpse = Attention(hidden_dim, use_logit_clip=False, use_cuda = use_cuda)
        self.use_cuda = use_cuda

        self.softmax = nn.Softmax()

        self.decoder_type = decoder_type

    def forward(self, decoder_input, embedding_x, hidden, encoder_output):
        '''
        Args:
        decoder_input (tensor): (batch_size, embedding_dim)
        embedding_x (tensor): (k, batch_size, embedding_dim)
        hidden (tuple): (h, c), initially,
                    (encoder_output[-1], encoder_output[-1]), (batch_size, hidden_dim)
        encoder_output (tensor): encoder output, shape is (k, batch_size, embedding_dim)

        Return:
        prob_list (list): list of probability through the sequence
        index_list (list): list of indice
        hidden (tuple): last layer hidden state and cell state
        '''
        batch_size = decoder_input.size(0)
        seq_len = embedding_x.size(0)
        # save result
        # Dummy and taxi are fixed in index 0 and 1
        prob_list = [torch.FloatTensor([[1] + [0] * (self.sequence_length - 1)]).to(decoder_input.device),
                     torch.FloatTensor([[0] + [1] + [0] * (self.sequence_length - 2)]).to(decoder_input.device)]
        index_list = [torch.LongTensor([0]).to(decoder_input.device), torch.LongTensor([1]).to(decoder_input.device)]  # dummy and taxi
        
        mask = torch.zeros(batch_size, seq_len).byte()
        if self.use_cuda:
            mask = mask.cuda()
        mask[[b for b in range(batch_size)], [0, 1]] = 1
        prev_idx = None

        embedding_x = embedding_x.permute(1, 0, 2) # to (batch_size, seq_len, embedding_dim)
        ref = encoder_output.permute(1, 2, 0) # to (batch_size, embedding_dim, seq_len)

        for i in range(2, self.sequence_length):
            h, c = self.lstm_cell(decoder_input, hidden)
            hidden = (h, c)
            g_l = h  # (batch_size, hidden_dim)
            for _ in range(self.num_glimpse):
                logit = self.glimpse(encoder_output, g_l)
                logit, mask = mask_logit(logit, mask, prev_idx)
                p = self.softmax(logit) # (batch_size, seq_len)
                # ref (batch_size, hidden_dim, seq_len)
                g_l = torch.bmm(ref, p.unsqueeze(2)).squeeze(2) # (batch_size, hidden_dim)
            logit = self.pointer(encoder_output, g_l)
            logit_mask, mask = mask_logit(logit, mask, prev_idx)

            if self.decoder_type == "greedy": # for validation
                probs = self.softmax(logit_mask) # batch_size, k
                prev_idx, decoder_input = greedy(probs, embedding_x)
            elif self.decoder_type == "sampling": # for training
                probs = self.softmax(softmax_temperature(logit_mask, T = 1.) )# batch_size, k
                prev_idx, decoder_input = sampling(probs, embedding_x, index_list)
            else:
                raise NotImplementedError
            # record previous index            
            index_list.append(prev_idx)
            
            # record probability (like lstm output)
            prob_list.append(probs)
        return prob_list, index_list    
    

class Embedding(nn.Module):
    def __init__(self, input_size, embedding_size, use_cuda = False):
        super().__init__()
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size))
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def forward(self, inputs):
        '''
        Args:
        inputs (batch_size, seq_len, input_size)

        Return:
        embedded (batch_size, seq_len, hidden_size)
        '''
        if self.use_cuda:
            inputs = inputs.cuda()
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        embedding = self.embedding.repeat(batch_size, 1, 1) # batch_size, input_size, embedding_size
        embedded = []
        for i in range(seq_len):
            # batch multiplication should be same dimensions, so, unsqueeze 1
            embedded.append(torch.bmm(inputs[:, i, :].unsqueeze(1).float(), embedding))
        embedded = torch.cat(embedded, 1)
        return embedded    
    

class PointerNet(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, 
                 seq_length, 
                 decoder_type = "sampling", num_glimpse = 1, use_cuda = False):
        super().__init__()

        # define encoder and decoder
        self.encoder = Encoder(embedding_dim, hidden_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, 
                               seq_length, 
                               decoder_type = decoder_type, 
                               num_glimpse = num_glimpse, use_cuda=use_cuda)

        decoder_input = torch.Tensor(embedding_dim).float()
        self.decoder_input = nn.Parameter(decoder_input) # trainable, default require grad is true
        self.decoder_input.data.uniform_(-1 / math.sqrt(embedding_dim), 1 / math.sqrt(embedding_dim))

        # embedding
        self.embedding = Embedding(input_dim, embedding_dim, use_cuda=use_cuda)

    def forward(self, x):
        '''
        propagate through the network
        Args:
        x (tensor): (batch_size, k, input_dim), like in paper, embedding dim = d

        Return:
        - prob_list (list): length is max length of decoder,
                with the element shape (batch_size, sequence length),
        - index_list (list): length is max length of decoder,
                with the element shape (batch_size,)
        '''
        batch_size = x.size(0)
        embedding_x = self.embedding(x).permute(1, 0, 2)
        encoder_output, (encoder_ht, encoder_ct) = self.encoder(embedding_x)

        # last layer output h, c as decoder initial state
        hidden = (encoder_ht.squeeze(0), encoder_ct.squeeze(0)) # (batch_size, hidden_dim)

        decoder_input = self.decoder_input.unsqueeze(0).repeat(batch_size, 1)
        prob_list, index_list = self.decoder(decoder_input, embedding_x, hidden, encoder_output)

        return prob_list, index_list
    

class Critic(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, process_iters, use_logit_clip = False, use_cuda = False):
        super().__init__()
        self.encoder = Encoder(embedding_dim, hidden_dim)
        self.process_block = Attention(hidden_dim, use_logit_clip=use_logit_clip, 
                                       use_cuda = use_cuda) # output is (batch_size, hidden_dim)
        self.decoder = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1)
                        )
        self.process_iters = process_iters
        self.softmax = nn.Softmax()
        self.embedding = Embedding(input_dim, embedding_dim, use_cuda = use_cuda)

    def forward(self, x):
        '''
        Args:
        x (tensor): with shape (k, batch_size, embedding_dim)

        Return:
        output (tensor): (batch_size, 1)
        '''
        batch_size = x.size(0)
        embedding_x = self.embedding(x).permute(1, 0, 2)
        encoder_output, (encoder_ht, encoder_ct) = self.encoder(embedding_x)
        ref = encoder_output.permute(1, 2, 0) # to (batch_size, embedding_dim, seq_len)
        g_l = encoder_ht.squeeze(0)
        for p in range(self.process_iters):
            logit = self.process_block(encoder_output, g_l)
            p = self.softmax(logit) # (batch_size, k)
            g_l = torch.bmm(ref, p.unsqueeze(2)).squeeze(2)
        output = self.decoder(g_l)
        return output    


class Model(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, seq_len, precedence, batch_size = 128 ,process_iters = 3, punishment: float = -1e5, use_cuda = False):
        super().__init__()
        self.pointer_net = PointerNet(input_dim, embedding_dim, hidden_dim, seq_len, use_cuda=use_cuda)
        self.critic = Critic(input_dim, embedding_dim, hidden_dim, process_iters, use_cuda=use_cuda)
        self.precedence = precedence
        self.punishment = punishment
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.use_cuda = use_cuda

    def forward(self, x):
        '''
        Args:
        x (batch_size, seq_len, input_dim)
        '''
        if self.use_cuda:
          x = x.cuda()
        prob_list, index_list = self.pointer_net(x)
        
        b = self.critic(x)

        pi = []
        probs = []
        for i, index in enumerate(index_list):

            pi_ = x[[j for j in range(self.batch_size)], index.data,:]
            pi.append(pi_)
            prob_ = prob_list[i]
            prob_ = prob_[[j for j in range(self.batch_size)], index.data]
            probs.append(prob_)

        L = tour_length(pi, use_cuda=self.use_cuda)
        L += infeasible_punishment(index_list, self.precedence, punishment=self.punishment)
        log_probs = 0
        for prob in probs:
            log_prob = torch.log(prob)
            log_probs += log_prob

        return L, log_probs, pi, index_list, b