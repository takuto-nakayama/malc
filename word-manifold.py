#  Word Manifold
#  computing Betti numbers of word n-grams. (cf. Fitz 2022)

#-----preprocess-----#
## importing modules
from classes import WordManifold
from datetime import datetime
import argparse, csv


## parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('lang', type=str)
parser.add_argument('text_path', type=str)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--window_size', type=int, default=5)
args = parser.parse_args()


## defining variables
text_path = args.text_path
lang = args.lang
if args.save_path == None:
    save_path = f'wm-{lang}-{datetime.now().microsecond}'
window_size = args.window_size

wm =WordManifold(lang, text_path, window_size)


#-----main processes-----#
ngram = wm.get_ngram()
skeleta = wm.get_skeleta(ngram)
B = wm.get_boundary(skeleta)
betti = wm.get_betti(B)


## saving results
with open(f'{save_path}', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(betti)
for n,b in enumerate(betti[1:]):
    print(f'n={n}: Betti={b}')
