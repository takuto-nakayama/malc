import argparse, classes, h5py, os


embedding = classes.EmbeddingModel()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lang', type=str, help='a language code of the in[ut data')
    parser.add_argument('token', type=str, help='token for which embeddings are to be extracted')
    parser.add_argument('encoded', type=dict, help='encoded sentences containing the token_ids')
    parser.add_argument('--batch', type=int, default=100, help='batch size for embedding extraction (default=100)')
    parser.add_argument('--model', type=str, default='bert-base-multilingual-cased', help='pretrained model name (default="bert-base-multilingual-cased")')
    parser.add_argument('--tokenizer', type=str, default='bert-base-multilingual-cased', help='pretrained tokenizer name (default="bert-base-multilingual-cased")')
    parser.add_argument('--save_path', type=str, default='../sample-data/embedding/', help='path to a directory to save the embedding data (default="../sample-data/embedding")')

    args = parser.parse_args()
    lang = args.lang
    token = args.token
    encoded = args.encoded
    batch = args.batch
    model = args.model
    tokenizer = args.tokenizer
    save_path = args.save_path

    embedding = classes.Embedding(
        model_name=model,
        tokenizer_name=tokenizer
        )
    
    embedding.embed(
        token=token,
        encoded=encoded,
        batch=batch
        )
    
    if f'embedding-{lang}.h5' not in os.listdir(save_path):
        with h5py.File(f'{save_path}/embedding-{lang}.h5', 'w') as f:
            f.creeate_dataset(
                name=token,
                data=embedding.output
                )
    else:
        with h5py.File(f'{save_path}/embedding-{lang}.h5', 'a') as f:
            f.create_dataset(
                name=token,
                data=embedding.output
                )