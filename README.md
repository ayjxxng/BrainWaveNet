# BrainWaveNet
A. Jeong*, D. Heo*, E. Kang, and H.-I. Suk, “BrainWaveNet: Wavelet-based Transformer for Autism Spectrum Disorder Diagnosis,” *In Proceedings of the Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Marrakesh, Morocco, Oct. 6-10, 2024.

<p align="center"><img src = "https://github.com/ayjxxng/BrainWaveNet/assets/113572871/ff727d13-f4d6-4013-9a85-9e397b9177b5" width="80%" height="80%"></p>

> **Abstract.** The diagnosis of Autism Spectrum Disorder (ASD) using resting-state functional Magnetic Resonance Imaging (rs-fMRI) is commonly analyzed through functional connectivity (FC) between Regions of Interest (ROIs) in the time domain. However, the time domain has limitations in capturing global information. To overcome this problem, we propose a wavelet-based transformer, BrainWaveNet, that leverages the frequency domain and learns spatial-temporal information for rs-fMRI brain diagnosis. Additionally, BrainWaveNet learns inter-relations between two different frequency-based features (real and imaginary parts) by cross-attention mechanisms, which allows for a deeper exploration of ASD. In our experiments using the ABIDE dataset, we validated the superiority of BrainWaveNet by comparing it with competing deep learning methods. Furthermore, we analyzed significant regions of ASD for neurological interpretation.

## Perfomance
**Dataset: ABIDE-I**
| Method | AUC | ACC (%) | SEN (%) | SPC (%) |
| :---: | :---: | :---: | :---: | :---: |
| BrainNetCNN  | 0.6907 ± 0.01 | 64.36 ± 1.19 | 53.40 ± 3.79 | **75.32 ± 2.31** |
| BrainNetTF   | 0.7147 ± 0.02 | 66.80 ± 1.78 | 61.06 ± 4.96 | 72.55 ± 8.08 |
| STAGIN (GARO)| 0.5886 ± 0.07 | 57.23 ± 5.69 | 57.02 ± 11.62 | 57.45 ± 7.41 |
| STAGIN (SERO)| 0.5839 ± 0.05 | 62.66 ± 3.54 | 51.92 ± 2.18 | 61.28 ± 4.02 |
| BoIT         | 0.6989 ± 0.02 | 62.66 ± 3.54 | 55.32 ± 5.53 | 70.00 ± 3.31 |
| BrainWaveNet (Ours) | **0.7388 ± 0.02** | **67.55 ± 2.04** | **66.49 ± 9.17** | 68.35 ± 9.80 |

**Dataset: ADHD-200**
| Method | AUC | ACC (%) | SEN (%) | SPC (%) |
| :---: | :---: | :---: | :---: | :---: |
| BrainNetCNN  | 0.6428 ± 0.01 | 62.06 ± 1.12 | 35.48 ± 4.56 | **82.36 ± 3.25** |
| BrainNetTF   | 0.6886 ± 0.03 | 63.71 ± 2.43 | 43.10 ± 13.71 | 79.45 ± 10.65 |
| STAGIN (GARO)| 0.5304 ± 0.04 | 51.04 ± 5.20 | 46.35 ± 32.98 | 54.77 ± 30.53 |
| STAGIN (SERO)| 0.5520 ± 0.04 | 57.18 ± 1.62 | 34.91 ± 20.96 | 74.28 ± 13.70 |
| BoIT         | 0.6748 ± 0.03 | 64.22 ± 3.04 | 49.52 ± 5.62 | 75.45 ± 1.11 |
| BrainWaveNet (Ours) | **0.7330 ± 0.01** | **65.98 ± 2.28** | **59.52 ± 12.49** | 70.91 ± 7.47 |

## Dependencies
- python 3.10.14
- pytorch 2.2.2
- torchvision 0.17.2
- torchaudio 2.2.2
- scikit-learn 1.4.1
- wandb 0.17.2
- hydra-core 1.3.2
