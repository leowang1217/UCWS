
import argparse
from train import get_model
from sgb.utils import *
import pickle




                
def load_model_and_vocab(args):
    vocab_path=args.model_dir+"/vocab"
    vocab=pickle.load(open(vocab_path,'rb'))
    print("vocab load")
    model_args=pickle.load(open("{}/config".format(args.model_dir),'rb'))
    model=get_model(model_args,vocab)
    state=torch.load("{}/{}".format(args.model_dir,args.checkpoint))
    model.load_state_dict(state)
    print("model load")
    return model

def get_test_data(args):
    test_data,test_meta=batch_sentences(paths=[args.test_set],
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    replace_special_symbols=args.replace_special_symbols,
                                    min_len=0,
                                    )
    print("test_data get",)
    return test_data,test_meta


                

def main(args):
    model=load_model_and_vocab(args)
    test_data,test_meta=get_test_data(args)
    predict=model.decode(test_data,test_meta)
    if args.postprocess_punct:
        postprocess_punct(predict)
    f= open(args.output_file,'w')
    for l in predict:
        f.write(l+'\n')
    f.flush()
    f.close()

    
if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_set',
                        help='Path to the test data',required=True)               
    parser.add_argument('--model_dir',
                        help='Path to the model dir', required=True)
    parser.add_argument('--output_file',
                        help='result', required=True)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of the batch.',
                        default=64)
    parser.add_argument('--checkpoint',
                        help='The size of the batch.',
                        default="checkpoint_last.pt")
    parser.add_argument('--replace_special_symbols',
                        help='Whether to replace_special_symbols.',
                        action="store_true")
    parser.add_argument('--postprocess_punct',
                        help='Whether to postprocess_punct.',
                        action="store_true")
    
    args = parser.parse_args()
    main(args)