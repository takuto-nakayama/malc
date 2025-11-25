import argparse, classes, h5py, os, pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lang', type=str, help='a language code of the in[ut data')
    parser.add_argument('token', type=str, help='token for which embeddings are to be extracted')
    parser.add_argument('path_encoded', type=str, help='path to encoded sentences containing the token_ids')
    parser.add_argument('--batch', type=int, default=100, help='batch size for embedding extraction (default=100)')
    parser.add_argument('--model', type=str, default='bert-base-multilingual-cased', help='pretrained model name (default="bert-base-multilingual-cased")')
    parser.add_argument('--tokenizer', type=str, default='bert-base-multilingual-cased', help='pretrained tokenizer name (default="bert-base-multilingual-cased")')
    parser.add_argument('--path_save', type=str, default='../sample-data/embedding', help='path to a directory to save the embedding data (default="../sample-data/embedding")')

    args = parser.parse_args()
    lang = args.lang
    token = args.token
    path_encoded = args.path_encoded
    batch = args.batch
    model = args.model
    tokenizer = args.tokenizer
    path_save = args.path_save

    embedding = classes.Embedding(
        model_name=model,
        tokenizer_name=tokenizer
        )
    
    with open(path_encoded, 'rb') as f:
        encoded = pickle.load(f)    

    embedding.embed(
        token=token,
        encoded=encoded,
        batch=batch
        )
    
    if lang not in os.listdir(path_save):
        os.makedirs(f'{path_save}/{lang}')

    if f'embedding-{lang}.h5' not in os.listdir(f'{path_save}/{lang}'):
        with h5py.File(f'{path_save}/{lang}/embedding-{lang}.h5', 'w') as f:
            f.creeate_dataset(
                name=token,
                data=embedding.output
                )
    else:
        with h5py.File(f'{path_save}/{lang}/embedding-{lang}.h5', 'a') as f:
            f.create_dataset(
                name=token,
                data=embedding.output
                )