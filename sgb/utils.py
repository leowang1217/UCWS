from typing import List
import torch
import random
import re
from tqdm import tqdm
PUNCT_DEFAULT=['＋', '∕', '（', '·', '{', '﹗', 'Γ', '：', '】', '◎', '}', '㎡', '〞', '﹄',
           'ㄍ', '》', '╳', '|', '〔', ']', 'ㄖ', '☆', '＝', 'Μ', '┘', 'Ο', '﹂', '$', 'Ν',
           '｀', '〉', '①', 'ㄒ', 'ㄧ', '=', '●', '；', '‘', 'ˋ', '＆', 'ㄇ', '～', 'ㄟ', '﹒', '﹕', '@', '※', '│', '﹃',
           'Τ', "'", '＊', 'β', 'ω', '、', '㈨', 'Ε', '×', '﹞', '㎝', '⑤', '﹔', '◢', '◇', ')',
           '｝', '／', '﹝', '＞', '］', 'Ⅰ', '’', '〈', '﹁', '﹑', '。', '+', '(', 'ˊ', ',',
           'Ⅲ', 'Ⅳ', 'μ', 'ⅰ', '，', 'λ', 'Β', 'Η', '?', 'Ι', '【', 'Ⅴ', '！', '㏄', '＇', '▲', 'Κ', '℃', '─', '㎜', '－', '⑥',
           '「', '≦', '˙', '┐', '⑸', '&', '°', '﹐', '→', '‧', '⑵', '［', '．', 'ⅱ', '＠', '』', '〇', '/', '②',  '」', '﹣',
           '“', '`', '？', 'Ⅹ', 'Ρ', '±', '◆', '＜', '★', 'ˇ', '『', '↑', '″', 'ㄑ', '<',
           '＼', '■', '”', '-', '⑦', '﹛', '③', '｛', 'Α', '＄', '□', '︰', '〕', '∶', 'Ⅱ', '《','…',
           '④', 'α', '﹖', '_', '.', '〝', 'ㄚ', '∥', '）', '△']

def batch_sentences(paths : List,
                    batch_size : int,
                    shuffle:bool,
                    replace_special_symbols : bool,
                    min_len:int =1,
                    max_len:int=5000):
    """
    batch the same length sentences
    """
    sentences = []
    symbols = []
    metadata={}
    for filename in paths:
        with open(filename,'r') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().replace('\u3000', ' ').replace(' ', '')
                if len(line) < min_len or int(len(line)) > max_len:
                    continue
                if replace_special_symbols:
                    tokens, eng, num = replace_num_and_letter(line)
                    sentences.append(tokens)
                    symbols.append((eng, num))
                else:
                    sentences.append(list(line))
    metadata["symbols"]=symbols
    lengths = [len(x) for x in sentences]
    metadata["perm"] = sorted(range(len(lengths)), key=lambda k: lengths[k])
    metadata["uperm"] = sorted(range(len(metadata["perm"])), key=lambda k: metadata["perm"][k])
    sentences.sort(key=lambda x: len(x))
    cur_batch_length = len(sentences[0])
    sent_batch = []
    batches = []
    for sent in sentences:
        if len(sent) == cur_batch_length:
            sent_batch.append(sent)
            if len(sent_batch) == batch_size:
                batches.append(sent_batch)
                sent_batch = []

        else:
            cur_batch_length = len(sent)
            if len(sent_batch) > 0:
                batches.append(sent_batch)
            sent_batch = [sent]
    if len(sent_batch) > 0:
        batches.append(sent_batch)
    if shuffle:
        random.shuffle(batches)
    
    return batches, metadata

def replace_num_and_letter(line):
    eng = iter(re.findall(r"[A-Za-zＡ-Ｚａ-ｚ]+",line))
    line=re.sub(r"[A-Za-zＡ-Ｚａ-ｚ]+",'♂',line)
    num = iter(re.findall(r"\d+\.?\d*", line))
    line=re.sub(r"\d+\.?\d*",'♀',line)
    tokens=[]
    for word in line:
        if word == '♀':
            word = '<NUM>'
        elif word == '♂':
            word = '<ENG>'
        tokens.append(word)
    return tokens, eng, num


def count_vocab(datas:List):
    voc = {'<PART>': 0, '<NUM>': 1, '<ENG>': 2, '<END>': 3, '<START>': 4,'<UNK>':5}
    for data in datas:
        for batch in tqdm(data):
            for sent in batch:
                for word in sent:
                    if word not in voc:
                        voc[word]=len(voc)
    return voc


def postprocess_punct(test_data,add_punct:List=[]):
    punct=PUNCT_DEFAULT
    regex = re.compile('\s+')
    if add_punct:
        punct.extend(add_punct)
    output=[]
    for line in test_data:
        line = regex.split(line.strip())
        out_sent=""
        for word in line:
            for char in word:
                if char in punct:
                    out_sent +=(' ' + char + ' ')
                else:
                    out_sent+=char
            out_sent+=' '
        output.append(out_sent)
    return output



def index_tokens(batch, word_to_ix):
    idx = [[word_to_ix[w] if w in word_to_ix else 5 for w in sent] for sent in batch]
    return torch.tensor(idx, dtype=torch.long, requires_grad=False).cuda()



def init_word(word, batch_size, word_to_ix):
        a = [word]
        sentence = [a for n in range(batch_size)]
        return index_tokens(sentence, word_to_ix).view(-1, 1)




def log_sum_exp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))

    return m + torch.log(sum_exp)



def combine_seq(seq_list):
    seq_set = set()
    for seq in seq_list:
        seq_set = seq_set | set(seq)
    a = list(seq_set)
    a.sort()
    return a

def seq2seq(seq_list):
    new_list = []
    for seq in seq_list:
        new_list.append(seq2sum(seq))
    new_seq = sum2seq(combine_seq(new_list))

    return new_seq




def seq2sum(seq):
    count = 0
    new_seq = []
    for number in seq:
        count += number
        new_seq.append(count)
    return new_seq


def sum2seq(sum):
    count = 0
    new_seq = []
    for x in sum:
        new_seq.append(x - count)
        count = x
    return new_seq

def _discretize(tokens):
    index = 0
    output = []
    for word in tokens:
        output.append(list(range(index,index+len(word))))
        index += len(word)
    return output



def get_f1(predict:List,gold:List):
    regex = re.compile('\s+')
    c,e,N = 0,0,0
    if len(predict)!=len(gold):
        print("ERROR")
        return
    for pl, gl in zip(predict,gold):
        gold_sent=_discretize(regex.split(gl.strip()))
        pred_sent=_discretize(regex.split(pl.strip()))
 
        for word in pred_sent:
            if word in gold_sent:
                c += 1
            else:
                e += 1
        N += len(gold_sent)
    recall = c / N
    precision = c / (c + e)
    f1 = (2 * recall * precision) / (recall + precision)
    er = e / N
    print(' r ', recall, ' p ', precision, ' f1 ', f1, ' er ', er)

def get_f1_from_file(predict_file,gold_file):
    regex = re.compile('\s+')
    c,e,N = 0,0,0
    predict = open(predict_file, 'r', encoding='utf-8').readlines()
    gold = open(gold_file ,'r', encoding='utf-8').readlines()
    if len(predict)!=len(gold):
        print("ERROR")
        return
    for pl, gl in zip(predict,gold):
        gold_sent=_discretize(regex.split(gl.strip()))
        pred_sent=_discretize(regex.split(pl.strip()))
 
        for word in pred_sent:
            if word in gold_sent:
                c += 1
            else:
                e += 1
        N += len(gold_sent)
    recall = c / N
    precision = c / (c + e)
    f1 = (2 * recall * precision) / (recall + precision)
    er = e / N
    print(' r ', recall, ' p ', precision, ' f1 ', f1, ' er ', er)

