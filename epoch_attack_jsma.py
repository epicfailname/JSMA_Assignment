import argparse
import os

import matplotlib.pyplot as plt
from keras.models import load_model

from attack_jsma import JSMARegressionAttack
from pixelmap import AlgorithmEnum
from utilities import SDC_data

parser = argparse.ArgumentParser(description="Attacking a Udacity SDC Regression CNN Model")
parser.add_argument('--modelsdir', type=str, help='directory to saved models', required=False)
parser.add_argument('--modelname', type=str, help='name of saved model to resume training on', required=False)
parser.add_argument('--maxiters', type=int, help='maximum number of iterations of pertubation', required=False)
parser.add_argument('--attackdir', type=str, help='use this argument if the attack info and images are in the same folder; supercede attackinfo and attackimagedir', required=True)
parser.add_argument('--resultsdir', type=str, help='name of the folder to save the results to', required=True)
parser.add_argument('--pixelmap', type=str, help='perform pixel mapping on the masks to perturb mapped pixels in parallel, selects mapping algorithm to use')

parser.add_argument('--debug', action='store_true', help='print debug messages?')
args = parser.parse_args()

PIXELMAP_ALGO = None
MODELS_DIR = './models'
MODEL_NAME = 'sdc-50epochs-shuffled'
IMAGE_FILE = None
IMAGE_FOLDER = None
RESULTS_DIR = None
MAX_ITERATIONS = 50

# Directory to saved models
if not args.modelsdir == None:
    MODELS_DIR = args.modelsdir

# Model name
if not args.modelsdir == None:
    MODEL_NAME = args.modelname
print('Loading a saved model:           ' + MODEL_NAME)

MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

if not args.maxiters == None:
    MAX_ITERATIONS = args.maxiters

# Path to attack folder
IMAGE_FILE = os.path.join(args.attackdir, 'attack.csv')
IMAGE_FOLDER = args.attackdir
print('Using attack target images from: ' + IMAGE_FOLDER)
print('Using attack target labels from: ' + IMAGE_FILE)

RESULTS_DIR = os.path.join('results/', args.resultsdir)
if not os.path.isdir(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

if args.pixelmap:
    # Set the enum for which algorithm to use
    if args.pixelmap not in ['homography']:
        raise ValueError('pixelmap is one of: "homography"')
    if args.pixelmap == 'homography':
        PIXELMAP_ALGO = AlgorithmEnum.HOMOGRAPHY
else:
    PIXELMAP_ALGO = None

data = SDC_data(IMAGE_FILE, IMAGE_FOLDER)
model = load_model(MODEL_PATH)

attack = JSMARegressionAttack(model,
        RESULTS_DIR,
        is_mask = True,
        pixelmap_algo = PIXELMAP_ALGO,
        max_iters = MAX_ITERATIONS,
        debug_flag=args.debug)

input_imgs = data.input_data
adv_imgs = attack.attack(data)
