import argparse, classes, os
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lang', type=str, help='a language code of the input data')
    parser.add_argument('token', type=str, help='token for which curvature is to be computed')
    parser.add_argument('data', type=np.ndarray, help='embeddings data as a numpy array')
    parser.add_argument('index', type=int, help='index of the point at which to compute the curvature')
    parser.add_argument('k', type=int, help='number of neighbors to consider')
    parser.add_argument('n', type=int, help='intrinsic dimension of the manifold')
    parser.add_argument('--save_path', type=str, default='../sample-data/curvature', help='path to save the curvature tensor (default="../sample-data/curvature")')

    args = parser.parse_args()
    lang = args.lang
    token = args.token
    data = args.data
    point = data[args.index]
    k = args.k
    n = args.n
    save_path = args.save_path
    tensor_id = f'{save_path}/{lang}-{k}-{n}'
    manifold = classes.Manifold()

    if f'{save_path}/{lang}-{token}-{k}-{n}' not in os.listdir(save_path):
        os.makedirs(f'{save_path}/{tensor_id}')

    for i in range(len(data)):
        R = manifold.curvature_tensor(data, point[i], k, n)

        with open(f'{save_path}/{tensor_id}/curvature-{lang}-{token}-{i}.npy', 'wb') as f:
            np.save(f, R)
