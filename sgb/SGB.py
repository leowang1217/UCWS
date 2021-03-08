import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from sgb.utils import *


class Encoder(nn.Module):
    def __init__(self,
                embedding_layer:nn.Embedding,
                vocab:dict,
                hidden_dim:int = 300):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_layer.embedding_dim
        self.lstm = nn.LSTM(embedding_layer.embedding_dim, hidden_dim, batch_first=True)
        self.embedding=embedding_layer
        self.vocab=vocab
        # self.part_tag = self.vocab()

    def forward(self, vector):
        batch_size = len(vector)
        part_tag = init_word('<PART>', batch_size, self.vocab)
        hidden = (torch.zeros(1,batch_size, self.hidden_dim).cuda(),
                       torch.zeros(1,batch_size, self.hidden_dim).cuda())
        embed = torch.cat((part_tag, vector), 1)
        out,hidden=self.lstm(self.embedding(embed),hidden)

        return torch.transpose(out,0,1)


class Decoder(nn.Module):
    # todo bidirection
    def __init__(self, embedding_layer:nn.Embedding,vocab:dict,hidden_dim:int =300):
        super(Decoder, self).__init__()
        # self.hidden_dim = hidden_dim
        self.embedding = embedding_layer
        # self.pre_lstm = nn.LSTMCell(embedding_layer.embedding_dim, hidden_dim)
        self.h2p = nn.Linear(hidden_dim, len(vocab))
        self.vocab=vocab 
        self.lstm = nn.LSTM(input_size=embedding_layer.embedding_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, target_index, pre_hidden):
        batch_size = len(target_index)
        seq_len = len(target_index[0])
        part_tag = init_word('<PART>', batch_size, self.vocab)
        lstm_input = self.embedding(torch.cat((part_tag, target_index), 1))
        pre_hidden = torch.unsqueeze(pre_hidden, 0)
        pre_hidden = (pre_hidden, pre_hidden)

        output, pre_hidden = self.lstm(lstm_input, pre_hidden)
        output = F.log_softmax(self.h2p(output), dim=2)
        select_character = torch.unsqueeze(target_index, 2)
        select_part_tag = torch.zeros((batch_size, seq_len, 1), dtype=torch.long, requires_grad=False).cuda()
        character_prob = torch.squeeze(output[:, :-1, :].gather(2, select_character), 2)
        tag_prob = torch.squeeze(output[:, 1:, :].gather(2, select_part_tag), 2)
        for n in range(1, seq_len):
            character_prob[:, n] += character_prob[:, n - 1]

        return torch.transpose((character_prob + tag_prob), 0, 1)

class SGB(nn.Module):
    def __init__(self,
                embedding_layer:nn.Embedding,
                vocab:dict,
                encoder_hidden_dim:int=300,
                decoder_hidden_dim:int=300,
                max_word_len:int=3,
                learning_rate:float=0.001
                ):
        super(SGB, self).__init__()
        self.penalty = -0.0
        self.connect = Encoder(embedding_layer,vocab,encoder_hidden_dim).cuda()
        self.forward_prob_model = Decoder(embedding_layer,vocab,decoder_hidden_dim).cuda()
        self.backward_prob_model = Decoder(embedding_layer,vocab,decoder_hidden_dim).cuda()
        self.max_word_len=max_word_len
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.vocab=vocab

    def forward(self, sentence):
        prob = self.get_prob_list(sentence)
        return self.compute_all_probability(prob, sentence)

    def get_prob_list(self, sentence):
        t = len(sentence[0])

        forward_vec = index_tokens(sentence, self.vocab)
        backward_vec = torch.flip(forward_vec, [1])

        forward_hidden_list = self.connect.forward(forward_vec[:, :-1])
        backward_hidden_list = self.connect.forward(backward_vec[:, :-1])
        backward_hidden_list = torch.flip(backward_hidden_list, [0])

        forward_prob = []
        backward_prob = []
        for m in range(0, t):
            n = m + self.max_word_len if m + self.max_word_len < t else t
            forward_prob.append(self.forward_prob_model(forward_vec[:, m:n], forward_hidden_list[m]))
            backward_prob.append(self.backward_prob_model(backward_vec[:, m:n], backward_hidden_list[t - m - 1]))

        return self.add_bidirection(forward_prob, backward_prob)



    def add_bidirection(self, forward_prob, backward_prob):

        return NotImplementedError


    def compute_all_probability(self, prob, sentence):
        
        dim=len(prob[0][0].size())
        total = [torch.zeros((len(sentence),dim), dtype=torch.float, requires_grad=False).squeeze().cuda()]
        t = len(sentence[0])
        for j in range(1, t + 1):
            low = 0 if j - self.max_word_len < 0 else j - self.max_word_len
            s = []
            for k in range(low, j):
                # print(total[k].size())
                # print(prob[j - 1][j - k - 1].size())
                s.append(total[k] + prob[j - 1][j - k - 1])
            p = log_sum_exp(torch.stack(s), dim=0)
            total.append(p)
        return total[t].mean() * -1

    def decode(self, test_data,metadata):
        return NotImplementedError


    def train_once(self, sentence):

        self.optimizer.zero_grad()
        loss = self.forward(sentence)
        loss.backward()
        self.optimizer.step()

        return loss.item() / len(sentence)


class SGB_A(SGB):
    def __init__(self,
                embedding_layer:nn.Embedding,
                vocab:dict,
                encoder_hidden_dim:int=300,
                decoder_hidden_dim:int=300,
                max_word_len:int=3,
                learning_rate:float=0.001
                ):
        super(SGB_A, self).__init__(
                embedding_layer,
                vocab,
                encoder_hidden_dim,
                decoder_hidden_dim,
                max_word_len,
                learning_rate)


    
    def add_bidirection(self, forward_prob, backward_prob):
        row = len(forward_prob)
        max_len = len(forward_prob[0])
        new_prob = []
        for x in range(row):
            m = x + 1 if x + 1 < max_len else max_len
            sub_prob = []
            for y in range(m):
                sub_prob.append((forward_prob[x - y][y] + backward_prob[-x - 1][y])/2)
            new_prob.append(sub_prob)
        return new_prob




    def decode(self, test_data,metadata):
        self.eval()
        print()
        symbols=metadata["symbols"]
        perm=metadata["perm"]
        uperm=metadata["uperm"]
        
        output = []
        # f = open(name, 'w', encoding='utf-8')
        with torch.no_grad():
            total_batch = len(test_data)
            cur_sent = 0

            for batch_index,batch in enumerate(test_data):
                sys.stdout.write('\r Decoding: {} / {}'.format(batch_index,total_batch))
                sys.stdout.flush()

                batch_size = len(batch)
                seq_len = len(batch[0])

                if seq_len == 0:
                    output.extend(['']*batch_size)
                    cur_sent += batch_size
                    continue


                prob = self.get_prob_list(batch)
                for sent_idx in range(batch_size):
                    max_prob= [torch.tensor(0, dtype=torch.float, requires_grad=False).cuda()]
                    argmax_prob = []
                    for j in range(1, seq_len + 1):
                            low = 0 if j - self.max_word_len < 0 else j - self.max_word_len
                            sent_prob = []
                            for k in range(low, j):
        
                                sent_prob.append(max_prob[k].view(-1) + prob[j - 1][j - k - 1][sent_idx].view(-1))
                            sent_prob = list(reversed(sent_prob))
                            sent_prob = torch.stack(sent_prob).view(-1)
                            max_prob.append(torch.max(sent_prob))
                            argmax_prob.append(torch.argmax(sent_prob))
                    seq = []
                    pos = seq_len - 1
                    while pos >= 0:
                            value = argmax_prob[pos].item() + 1
                            seq.append(value)
                            pos = pos - (value)

                    seq = list(reversed(seq))
                    
                    if symbols:
                        eng, num = symbols[perm[cur_sent]]  
                    char_idx = 0
                    target_str = ''
                    for word_len in seq:
                        for n in range(word_len):
                            char = batch[sent_idx][char_idx]
                            char_idx += 1
                            if char == '<NUM>':
                                char = next(num)
                            elif char == '<ENG>':
                                char = next(eng)
                            target_str += char
                        target_str += ' '
                    cur_sent += 1
                    output.append(target_str)
            return [output[idxs] for idxs in uperm]


            
class SGB_C(SGB):
    def __init__(self,
                embedding_layer:nn.Embedding,
                vocab:dict,
                encoder_hidden_dim:int=300,
                decoder_hidden_dim:int=300,
                max_word_len:int=3,
                learning_rate:float=0.001
                ):
        super(SGB_C, self).__init__(
                embedding_layer,
                vocab,
                encoder_hidden_dim,
                decoder_hidden_dim,
                max_word_len,
                learning_rate)


    
    def add_bidirection(self, forward_prob, backward_prob):
        row = len(forward_prob)
        max_len = len(forward_prob[0])
        new_pro = []
        for x in range(row):
            m = x + 1 if x + 1 < max_len else max_len
            sub_pro = []
            for y in range(m):
                sub_pro.append(torch.stack((forward_prob[x - y][y], backward_prob[-x - 1][y]), 1))
            new_pro.append(sub_pro)
        return new_pro




    def decode(self, test_data,metadata):
        self.eval()
        print()
        symbols=metadata["symbols"]
        perm=metadata["perm"]
        uperm=metadata["uperm"]
        
        output = []
        # f = open(name, 'w', encoding='utf-8')
        with torch.no_grad():
            total_batch = len(test_data)
            cur_sent = 0

            for batch_index,batch in enumerate(test_data):
                sys.stdout.write('\r Decoding: {} / {}'.format(batch_index,total_batch))
                sys.stdout.flush()

                batch_size = len(batch)
                seq_len = len(batch[0])

                if seq_len == 0:
                    output.extend(['']*batch_size)
                    cur_sent += batch_size
                    continue

                
                prob = self.get_prob_list(batch)
                argmax_probs=[]
                for direction in range(2):

                    max_prob = [torch.zeros(batch_size, dtype=torch.float, requires_grad=False).cuda()]
                    argmax_prob = []
                    for j in range(1, seq_len + 1):
                            low = 0 if j - self.max_word_len < 0 else j - self.max_word_len
                            probs = []
                            for k in range(low, j):
                                probs.append(max_prob[k] + prob[j - 1][j - k - 1][:, direction])
                            probs = list(reversed(probs))
                            probs = torch.stack(probs, 0)
                            probs = torch.max(probs, 0)
                            max_prob.append(probs[0])
                            argmax_prob.append(probs[1])

                    argmax_prob = torch.stack(argmax_prob, 0)
                    argmax_probs.append(argmax_prob)
                for sent_idx in range(batch_size):
                    seq_list = []
                    for direction in range(2):
                        pos = seq_len - 1
                        seq = []
                        while pos >= 0:
                            value = argmax_probs[direction][pos][sent_idx].item() + 1
                            seq.append(value)
                            pos = pos - value

                        seq = list(reversed(seq))
                        seq_list.append(seq)
                    seq = seq2seq(seq_list)
               
                    if symbols:
                        eng, num = symbols[perm[cur_sent]]  
                    char_idx = 0
                    target_str = ''
                    for word_len in seq:
                        for n in range(word_len):
                            char = batch[sent_idx][char_idx]
                            char_idx += 1
                            if char == '<NUM>':
                                char = next(num)
                            elif char == '<ENG>':
                                char = next(eng)
                            target_str += char
                        target_str += ' '
                    cur_sent += 1
                    output.append(target_str)
            return [output[idxs] for idxs in uperm]           
