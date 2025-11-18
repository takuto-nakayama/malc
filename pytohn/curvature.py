import argparse
import classes
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=np.ndarray, help='embeddings data as a numpy array')
    parser.add_argument('index', type=int, help='index of the point at which to compute the curvature')
    parser.add_argument('k', type=int, help='number of neighbors to consider')
    parser.add_argument('n', type=int, help='intrinsic dimension of the manifold')
    args = parser.parse_args()
    
    data = args.data
    point = data[args.index]
    k = args.k
    n = args.n
    manifold = classes.Manifold()

    g, dg,J, H = manifold.metric(data, point, k, n)
    gamma, dgamma = manifold.christoffel(g, dg, J, H)
    R = manifold.riemann(gamma, dgamma)