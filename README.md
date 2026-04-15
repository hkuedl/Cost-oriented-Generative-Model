# Cost-oriented-Generative-Model

This repository contains the code and related materials for the paper **“Cost-oriented Scenario Generation for Power Systems Under Uncertainty”**, which has been submitted to *IEEE Transactions on Smart Grid*.

**Authors:** Yangze Zhou, Yihong Zhou, Daniel Kirschen, Thomas Morstyn, and Yi Wang.

**Corresponding author:** Yi Wang (<yiwang@eee.hku.hk>)

## Data
Data can be found in [Google Drive Folder](https://drive.google.com/drive/folders/1ArEguiHyZMoi7oXKkbaTTZG9JtMWypMH?usp=drive_link)

## Environment

We recommend using Conda to manage the Python environment for this project.

Create the environment from the provided file:

```bash
conda env create -f environment.yml
conda activate Meta_DFL
```

## How to Run

<table>
  <thead>
    <tr>
      <th>Training Setting</th>
      <th>Model</th>
      <th>How to Run</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Standard generative model</td>
      <td>VAE</td>
      <td>Run the notebooks <code>forecasting_VAE_joint.ipynb</code> and <code>forecasting_VAE_separate.ipynb</code></td>
    </tr>
    <tr>
      <td>GAN</td>
      <td>Run the notebooks <code>forecasting_GAN_joint.ipynb</code> and <code>forecasting_GAN_separate.ipynb</code></td>
    </tr>
    <tr>
      <td>Diffusion</td>
      <td>Run the notebooks <code>forecasting_diffusion_joint.ipynb</code> and <code>forecasting_diffusion_separate.ipynb</code></td>
    </tr>
    <tr>
      <td rowspan="3">Cost-oriented generative model</td>
      <td>VAE</td>
      <td>Run the commands <code>python main_VAE_joint.py</code> and <code>python main_VAE_separate.py</code></td>
    </tr>
    <tr>
      <td>GAN</td>
      <td>Run the commands <code>python main_GAN_joint.py</code> and <code>python main_GAN_separate.py</code></td>
    </tr>
    <tr>
      <td>Diffusion</td>
      <td>Run the commands <code>python main_diffusion_joint.py</code> and <code>python main_diffusion_separate.py</code></td>
    </tr>
  </tbody>
</table>

## Results

Because the forecasting model parameters are too large to be uploaded to GitHub, they are provided separately via Google Drive.

Please note that this repository currently contains only partial experimental results. For the complete version, including the full model parameters and related files, please visit:

[Google Drive Folder](https://drive.google.com/drive/folders/1ArEguiHyZMoi7oXKkbaTTZG9JtMWypMH?usp=drive_link)

