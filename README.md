# Overview

This repository holds the data and scripts to replicate the analysis and outputs in the forthcoming paper, 
["Satellite monitoring uncovers progress but large disparities in doubling crop yields."](https://arxiv.org/abs/2411.03322) Please do include attribution
if using the data or analysis: 

```yaml
@misc{fankhauser2024satellitemonitoring,
      title={Satellite monitoring uncovers progress but large disparities in doubling crop yields}, 
      author={Katie Fankhauser and Evan Thomas and Zia Mehrabi},
      year={2024},
      eprint={2411.03322},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2411.03322), 
}
```

# Setup

Download this repository to a project directory on your local computer: 

```
git clone https://github.com/Better-Planet-Laboratory/sdg-maize.git
cd sdg-maize
```

The `environment.yml` file includes all required packages needed to set up your
python environment for this project, in conda for instance: 

```
conda env create -f environment.yml
conda activate sdg_maize
```

# Data

The aggregate data needed to run the analysis -- see [`scripts/manuscript.py`](./scripts/manuscript.py) -- 
are included as part of this repo in the [`data`](./data) directory.

The source data and machine learning pipeline to predict seasonal maize cover and yields at high resolution is described in: 

Fankhauser K, Thomas E, Brook C, Gatera A, Mehrabi Z. High-resolution
wall-to-wall time series of seasonal maize area and yield for Rwanda over
2019–2023. Environmental Research: Food Systems. 2025 sep;2(4):045003. 
[https://doi.org/10.1088/2976-601X/ae033c](https://doi.org/10.1088/2976-601X/ae033c).

with the scripts and 10m raster data available for download at:
[https://zenodo.org/records/10659095](https://zenodo.org/records/10659095)
