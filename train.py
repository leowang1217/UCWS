
import argparse
import os
import shutil
from sgb.SGB import SGB_A, SGB_C
import sys
import numpy as np
import torch.nn as nn
from sgb.utils import *
import pickle

def get_data(args):
    
    train_data,_=batch_sentences(paths=args.train_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                replace_special_symbols=args.replace_special_symbols,
                                min_len=args.min_len,
                                max_len=args.max_len,
                                )
    
    print("data get")
    # print(train_data[0][0])
    return train_data

def get_vocab(args,data):

    vocab_path=args.model_dir+"/vocab"
    if os.path.exists(vocab_path):
        vocab=pickle.load(open(vocab_path,'rb'))
    else:
        vocab=count_vocab(data)
        pickle.dump(vocab,open(vocab_path,'wb'))
    print("vocab get")
    return vocab

    


def get_model(args,vocab):


    embedding_layer = nn.Embedding(len(vocab), args.embedding_dim)
    pre_word_embedding = np.random.normal(0, 1, size=embedding_layer.weight.size())
    embedding_layer.weight.data.copy_(torch.from_numpy(pre_word_embedding))
    
    if args.model=="sgb-c":
        model=SGB_C(embedding_layer=embedding_layer,
                    vocab=vocab,
                    encoder_hidden_dim=args.encoder_hidden_dim,
                    decoder_hidden_dim=args.decoder_hidden_dim,
                    max_word_len=args.max_word_len,
                    learning_rate=args.lr
        )
    else:
        model=SGB_A(embedding_layer=embedding_layer,
                    vocab=vocab,
                    encoder_hidden_dim=args.encoder_hidden_dim,
                    decoder_hidden_dim=args.decoder_hidden_dim,
                    max_word_len=args.max_word_len,
                    learning_rate=args.lr
        )
    model=model.cuda()
    print("model get")
    return model
    
def trainer(args,model,train_data):
        model.train()
        pickle.dump(args,open("{}/config".format(args.model_dir),'wb'))
        for e in range(args.n_epoch):
            loss = 0
            random.shuffle(train_data)
            total=len(train_data)
            for index, sentence in enumerate(train_data):
                loss = model.train_once(sentence)
                # ave_loss=loss/(index+1)
                sys.stdout.write('\r  epoch: {} , {} / {} , loss: {}'.format(e,index,total,round(loss, 9)))
                sys.stdout.flush()
            print()
            
            torch.save(model.state_dict(), "{}/checkpoint{}.pt".format(args.model_dir,e))
            shutil.copy("{}/checkpoint{}.pt".format(args.model_dir,e),"{}/checkpoint_last.pt".format(args.model_dir))

            
                

                

def main(args):
    train_data=get_data(args)
    vocab=get_vocab(args,[train_data])
    model=get_model(args,vocab)
    trainer(
        args=args,
        model=model,
        train_data=train_data
    )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', metavar='N', nargs='+',
                        help='Path to the training data',required=True)             
    parser.add_argument('--model_dir',
                        help='Path to the model dir', required=True)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of the batch.',
                        default=64)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length',
                        default=5000)
    parser.add_argument('--encoder_hidden_dim',
                        type=int,
                        default=300)
    parser.add_argument('--decoder_hidden_dim',
                        type=int,
                        default=300)
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=300)
    parser.add_argument('--min_len',
                        type=int,
                        help='The min sentence length',
                        default=1)
    parser.add_argument('--max_word_len',
                        type=int,
                        help='The max word length',
                        default=3)
    parser.add_argument('--n_epoch',
                        type=int,
                        help='The number of epoch for training model.',
                        default=5)
    parser.add_argument('--lr',
                        type=float,
                        help='Set initial learning rate.',
                        default=1e-3)

    parser.add_argument('--replace_special_symbols',
                        help='Whether to replace_special_symbols.',
                        action="store_true")

    parser.add_argument('--model',
                        choices=['sgb-a', 'sgb-c'],
                        help='Name of the segmentation model.',
                        default='sgb-c')
    
    args = parser.parse_args()
    main(args)