# HYPER
A physics-conditioned, self-supervised reconstruction network for Fourier light-field microscopy

===========================================================================================
1. Install CUDA 12.6:
https://developer.nvidia.com/cuda-toolkit-archive

===========================================================================================
2. Install Anaconda:
https://www.anaconda.com/download

===========================================================================================
3. CLONE REPOSITORY
git clone https://github.com/ShuJiaLab/HYPER.git

===========================================================================================
4. CREATE ENVIRONMENT
conda env create -f environment.yml

===========================================================================================
5. ACTIVATE ENVIRONMENT
conda activate .conda

===========================================================================================
6. VERIFY PYTHON VERSION
python --version
Expected: Python 3.11

===========================================================================================
7. PREPARE DATA
Download dataset manually from:
>>> https://figshare.com/articles/dataset/Training_data_sets_containing_500_synthetic_light-field_data/32160783 <<<
and PSF data from:
>>> https://figshare.com/articles/dataset/PSF_data_for_HYPER_training/32162094 <<<
Then update the dataset paths in the XML configuration file 
located under /trainingparameters/

Open the XML file (e.g., trainingparameters/useWFsynLFdeep_20251207_16X.xml)
and modify the following fields to match your local directories:

<data_folder>./data/training/</data_folder>
<data_folder_test>./data/test/</data_folder_test>

and update the psf paths in the training code

===========================================================================================
8. RUN TRAINING
If you renamed the XML file, update the filename accordingly in your command
Then run:
python mainTrainFLFMnet.py

During training, logs are saved for visualization with TensorBoard.
Open your browser and go to:
http://localhost:6006/ (this should match the path defined in your training code)

===========================================================================================
9. TRAINING OUTPUT
Training outputs will be saved to:
./checkpoints/

===========================================================================================
10. Inference

1. Edit testparameters XML:
<data_folder type="str">/your/test/data/</data_folder>

2. Update file names in mainReconFLFMnet.py

3. Run:
python mainReconFLFMnet.py

Results will be saved under:
./checkpoints/
