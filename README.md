# Survival Reinforced Cohort Adaptation for Proteomic Subtyping and Significant Protein Discovery with Model Interpretation

This is the source code and data for reproducing the results and figures of the papar with hte same title.

1. Create a conda environment for tensorflow

conda create -n srps_env python=3.7 tensorflow-gpu=2.2 numpy scikit-learn scipy pandas progressbar2 statsmodels
conda activate srps_env
conda install -c conda-forge matplotlib lifelines matplotlib-venn
conda install -c bioconda harmonypy gseapy
conda install -c numba numba

2. Create a conda environment for R

conda create -n r_env
conda activate r_env
conda install -c conda-forge r-base=4.1.0 r-essentials=4.1
Rscript scripts_r/install_packages.R
*Note that you need to set 'repos_http' to a suitable CRAN mirror address according to your region in the script. Avaible mirrors are listed in https://cran.r-project.org/mirrors.html


3. Download data from https://drive.google.com/file/d/1pe9SorTIExJqCO0XQOpgp3ZUkgmAkG97/view?usp=sharing to the SRPS folder.
cd PATH_TO_THE_SRPS_FOLDER
tar -zf data.tar

4. Run experiment 

conda activate srps_env
bash cmd1.sh
bash cmd2.sh
bash cmd3.sh
bash cmd4.sh
bash cmd5.sh

5. run GSEA with r

conda activate r_env
Rscript scripts_r/CalculateDEP_ssGSEA_Jiang2Gao.R
Rscript scripts_r/CalculateDEP_ssGSEA_Jiang2Xu.R

6. Calculate metrics

conda activate srps_env
python scripts_py/train_models.py --mode compare --data toy --experiment toy
python scripts_py/train_models.py --mode compare -batch_effect lv1 --data synthetic --experiment synthetic
python scripts_py/train_models.py --mode compare -batch_effect lv3 --data synthetic --experiment synthetic
python scripts_py/train_models.py --mode compare --data HCC --experiment Jiang2Gao
python scripts_py/train_models.py --mode compare --data HCC_LUAD --experiment Jiang2Xu

7. Plot figures

python scripts_py\plotting.py --fig_name toy_test --data toy --experiment toy
python scripts_py\plotting.py --fig_name benchmarking_synthetic --data synthetic --experiment synthetic
python scripts_py\plotting.py --fig_name benchmarking_real --data HCC --experiment Jiang2Gao
python scripts_py\plotting.py --fig_name benchmarking_real --data HCC_LUAD --experiment Jiang2Xu
python scripts_py\plotting.py --fig_name model_explaination
