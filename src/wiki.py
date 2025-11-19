#  Script to process Wikipedia data and store it in HDF5 format.

## Importing necessary libraries
from transformers import BertTokenizer, BertModel
import argparse
import classes
import h5py


if __name__ == '__main__':
    ## Setting up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('lang', type=str, help='wiki code of a language)')
    parser.add_argument('token', type=str, help='token for which sentences are to be extracted')
    parser.add_argument('--tokenizer', type=str, default='bert-base-multilingual-cased', help='pretrained tokenizer name (default="bert-base-multilingual-cased")')
    parser.add_argument('--model', type=str, default='bert-base-multilingual-cased', help='pretrained model name (default="bert-base-multilingual-cased")')
    parser.add_argument('--range', type=int, nargs=2, default=[0,10000], help='range of texts from Wikipedia dataset (default=[0,10000])')
    parser.add_argument('--batch', type=int, default=1000, help='batch size for processing texts (default=1000)')

    args = parser.parse_args()
    lang = args.lang
    token = args.token
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = BertModel.from_pretrained(args.model)
    range = args.range
    batch = args.batch

    wiki = classes.Wiki(lang=lang)
    n_batch = (range[1]-range[0]) // batch + 1


    ## Main processing
    for _ in range(1, n_batch):
        print(f'\n\nProcessing: batch number is {_} ({_*batch}/{range[1]-range[0]} texts)')
        wiki.get_sentence(token=token, text_range=(range[0]+(_-1)*batch, min(range[0]+_*batch, range[1])))
        wiki.filtered
