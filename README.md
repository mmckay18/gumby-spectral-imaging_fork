# gumby-spectral-imaging
Contains code for the experiments in:
- [Quantifying the robustness of deep multispectral segmentation models against natural perturbations and data poisoning](https://arxiv.org/abs/2305.11347): MSI_robustness  
- [Impact of architecture on robustness and interpretability of multispectral deep neural networks](https://arxiv.org/abs/2309.12463): MSI_interpretability  

## Environment
In general, conda environments were generated using miniconda for CUDA11.4, python 3.10, torch 1.12, and rasterio 1.3. A conda environment is specified in `gby.yml`. In theory it can be created using the following command:  

``` bash
conda env create --file=environment.yml 
```

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{10.1117/12.2663498,
author = {Elise Bishoff and Charles Godfrey and Myles McKay and Eleanor Byler},
title = {{Quantifying the robustness of deep multispectral segmentation models against natural perturbations and data poisoning}},
volume = {12519},
booktitle = {Algorithms, Technologies, and Applications for Multispectral and Hyperspectral Imaging XXIX
},
editor = {Miguel Velez-Reyes and David W. Messinger},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {125190M},
keywords = {deep learning, robustness, multimodal, adversarial machine learning, data poisoning, natural robustness, multispectral },
year = {2023},
doi = {10.1117/12.2663498},
URL = {https://doi.org/10.1117/12.2663498}
}

@inproceedings{10.1117/12.2662998,
author = {Charles Godfrey and Elise Bishoff and Myles McKay and Eleanor Byler},
title = {{Impact of model architecture on robustness and interpretability of multispectral deep learning models}},
volume = {12519},
booktitle = {Algorithms, Technologies, and Applications for Multispectral and Hyperspectral Imaging XXIX
},
editor = {Miguel Velez-Reyes and David W. Messinger},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {125190L},
keywords = {Deep learning, multispectral images, multimodal fusion, robustness, interpretability},
year = {2023},
doi = {10.1117/12.2662998},
URL = {https://doi.org/10.1117/12.2662998}
}
```
