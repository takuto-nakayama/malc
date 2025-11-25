#  Script to process Wikipedia data and store it in HDF5 format.

## Importing necessary libraries
from transformers import BertTokenizer, BertModel
import argparse, classes, pickle


if __name__ == '__main__':
    ## Setting up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('lang', type=str, help='wiki code of a language)')
    parser.add_argument('token', type=str, help='token for which sentences are to be extracted')
    parser.add_argument('--tokenizer', type=str, default='bert-base-multilingual-cased', help='pretrained tokenizer name (default="bert-base-multilingual-cased")')
    parser.add_argument('--model', type=str, default='bert-base-multilingual-cased', help='pretrained model name (default="bert-base-multilingual-cased")')
    parser.add_argument('--num_text', type=int, default=10000, help='the number of texts from Wikipedia dataset (default=10000)')
    parser.add_argument('--save_path', type=str, default='../sample-data/encoded/', help='path to a directory to save the encoded data (default="../sample-data/encoded")')

    args = parser.parse_args()
    lang = args.lang
    token = args.token
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = BertModel.from_pretrained(args.model)
    num_text = args.num_text
    save_path = args.save_path

    wiki = classes.Wiki(lang=lang)


    ## Main processing
    wiki.get_sentence(token=token, num_text=num_text)
    
    with open(f'{save_path}/encoded-wiki-{lang}-{token}.pkl', 'wb') as f:
        pickle.dump(wiki.filtered, f)
