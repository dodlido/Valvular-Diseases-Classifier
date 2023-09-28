from pre_processing.pre_process import load_directory_top
from segmentation.correlation import estimate_dataset
from segmentation.labeling import guess_dataset
from classification.builddataset import create_dataset
from classification.features_extraction import extract_features_top
from classification.classifiers import classifier_top
from classification.classifiers import classifier_combined
from utils.utils import *
import warnings
warnings.filterwarnings("ignore")

RAW, PROC, SEG, FEAT, RES = 0, 1, 2, 3, 4
SAN1, SAN2, PETER, COMB = 0, 1, 2, 3
MAN, MELI, MELV = 0, 1, 2
SR = 1

root_dir = 'D:/datasets/Heartsounds/'  #  Change this
paths_txt = root_dir + 'paths_txt.txt'  
params_txt = root_dir + 'params_txt.txt'  

paths_stage, paths_dataset, paths_featype = unpack_paths_txt(paths_txt)
load_parameters = unpack_parameters(params_txt)


# iterate over all datasets:
for data_ind in range(COMB):

    # 0. define paths to current directories:
    dataset_name, current_stage = unpack_stage(root_dir, paths_dataset, paths_stage, data_ind)
          
    # 1. pre-process raw wavs:
    load_directory_top(current_stage[RAW], current_stage[PROC], dataset_name, load_parameters, current_stage[RES])
    
    # 2. estimate heartrates for each recording:
    estimate_dataset(current_stage[PROC])

    # 3. guess S1 locations for each recording:
    guess_dataset(current_stage[PROC], current_stage[RES])

    # 4. segments each recording to cycles according to guesses:
    create_dataset(current_stage[PROC], current_stage[SEG])

    # 5. extract features for each cycle:
    extract_features_top(current_stage[SEG], current_stage[FEAT], paths_featype, load_parameters[SR], current_stage[RES])

    # 6. classify each cycle, save results:
    classifier_top(current_stage[FEAT], current_stage[RES], paths_featype, dataset_name)

classifier_combined(root_dir, paths_stage[FEAT], paths_dataset, paths_featype, paths_stage[RES])
