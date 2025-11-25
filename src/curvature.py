import argparse, classes, h5py, os
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lang', type=str, help='a language code of the input data')
    parser.add_argument('token', type=str, help='token for which curvature is to be computed')
    parser.add_argument('path_data', type=np.ndarray, help='path to embeddings data as a numpy array')
    parser.add_argument('index', type=int, help='index of the point at which to compute the curvature')
    parser.add_argument('k', type=int, help='number of neighbors to consider')
    parser.add_argument('n', type=int, help='intrinsic dimension of the manifold')
    parser.add_argument('--path_save', type=str, default='../sample-data/curvature', help='path to save the curvature tensor (default="../sample-data/curvature")')

    args = parser.parse_args()
    lang = args.lang
    token = args.token
    path_data = args.path_data
    k = args.k
    n = args.n
    path_save = args.path_save
    id_tensor = f'{path_save}/{lang}-{k}-{n}'
    manifold = classes.Manifold()

    if f'{path_save}/{lang}-{token}-{k}-{n}' not in os.listdir(path_save):
        os.makedirs(f'{path_save}/{id_tensor}')

    with h5py.File(args.data, 'r') as f:
        data = f['embeddings'][:]

        for i in range(len(data)):
            R = manifold.curvature_tensor(data, data[i], k, n)

            with open(f'{path_save}/{id_tensor}/curvature-{lang}-{token}-{i}.npy', 'wb') as f:
                np.save(f, R)
