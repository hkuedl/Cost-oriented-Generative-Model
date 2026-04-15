# Cost-oriented-Generative-Model

This repository contains the code and related materials for the paper **“Cost-oriented Scenario Generation for Power Systems Under Uncertainty”**, which has been submitted to *IEEE Transactions on Smart Grid*.

**Authors:** Yangze Zhou, Yihong Zhou, Daniel Kirschen, Thomas Morstyn, and Yi Wang.

**Corresponding author:** Yi Wang (<yiwang@eee.hku.hk>)

## Data
Data can be found in "./Data/load_data_city_4_2.csv"

## Environment

We recommend using Conda to manage the Python environment for this project.

Create the environment from the provided file:

```bash
conda env create -f environment.yml
conda activate Meta_DFL

## How to Run

| Training Setting | Model | How to Run |
|---|---|---|
| Standard generative model training | VAE | Run the notebooks `forecasting_VAE_joint.ipynb` and `forecasting_VAE_separate.ipynb` |
| Standard generative model training | GAN | Run the notebooks `forecasting_GAN_joint.ipynb` and `forecasting_GAN_separate.ipynb` |
| Standard generative model training | Diffusion | Run the notebooks `forecasting_diffusion_joint.ipynb` and `forecasting_diffusion_separate.ipynb` |
| Cost-oriented generative model training | VAE | Run the commands `python main_VAE_joint.py` and `python main_VAE_separate.py` |
| Cost-oriented generative model training | GAN | Run the commands `python main_GAN_joint.py` and `python main_GAN_separate.py` |
| Cost-oriented generative model training | Diffusion | Run the commands `python main_diffusion_joint.py` and `python main_diffusion_separate.py` |


## Results

Because the forecasting model parameters are too large to be uploaded to GitHub, they are provided separately via Google Drive.

Please note that this repository currently contains only partial experimental results. For the complete version, including the full model parameters and related files, please visit:

[Google Drive Folder](https://drive.google.com/drive/folders/1ArEguiHyZMoi7oXKkbaTTZG9JtMWypMH?usp=drive_link)

