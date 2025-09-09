# Siamese Network Embeddings for Brain Region Ranking in MRI Twin Identification

[![Models](https://img.shields.io/badge/Models-Available-blue)](https://drive.google.com/drive/folders/1AT22UDsgiR6NRpqpN0CBJRxVZWEFfX68?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep learning framework for identifying genetic similarity in brain MRI data using Siamese networks with Layer-Wise Relevance Propagation (LRP) analysis.

## Overview

- **Identifies monozygotic twins** from brain MRI scans using 3D CNN Siamese networks
- **Ranks brain regions** by discriminative importance for genetic relatedness detection  
- **Provides interpretable results** through Layer-Wise Relevance Propagation analysis

## Dataset

- **Source**: Human Connectome Project S1200 Data Release
- **Subjects**: 138 monozygotic twin pairs (276 subjects)
- **Resolution**: T1-weighted images (260 × 311 × 260 voxels)
- **Processing**: Downscaled to 86 × 103 × 86 for training

![Dataset Example](assets/dataset_example.png)

<p align="center">
<i>Left to right: axial, coronal, and sagittal views of a random HCP T1-weighted \gls{mri} scan from the HCP 1200 Subjects Data Release preprocessed using the HCP minimal preprocessing pipelines, showing the 260 × 311 × 260 voxel dimensions</i>
</p>

## Architecture

- **Three 3D CNN backbones**: Modified U-Net, ResNet-18, and DenseNet-121
- **Siamese network design** with 128-dimensional embeddings
- **Triplet loss optimization** with hard negative mining
- **LRP attribution analysis** for spatial interpretation

![Average LRP](assets/averagelrp.png)

<p align="center">
<i>Reference LRP maps averaged across all subjects and embedding dimensions for each architecture (left: axial, middle: coronal, right: sagittal views). All models converge on similar brain regions with consistent focus on subcortical structures and brainstem. Color scale (0.0-1.0) represents normalized LRP attribution values, where higher values indicate regions contributing more to embedding representation.</i>
</p>


## Performance

| Architecture | Accuracy ↑ | F1-Score ↑ | AUC-ROC ↑ | Precision ↑ | Recall ↑ |
|--------------|------------|------------|-----------|-------------|----------|
| **U-Net**    | **91.4 ± 2.9** | **92.0 ± 2.5** | **95.2 ± 2.3** | 87.1 ± 5.1 | **97.9 ± 2.9** |
| ResNet-18    | 89.6 ± 2.2 | 89.6 ± 2.1 | 92.8 ± 2.4 | **90.8 ± 4.7** | 88.6 ± 3.5 |
| DenseNet-121 | 87.1 ± 4.8 | 88.5 ± 3.6 | 91.9 ± 3.1 | 81.6 ± 6.5 | 97.1 ± 1.4 |

<p align="center">
<i>Performance metrics showing mean ± standard deviation across 10 evaluation runs with different randomly generated combinations of non-twin pairs to assess robustness to negative sample selection. Bold values indicate best performance per metric.</i>
</p>

![Loss Graphs](assets/lossgraphs.png)

<p align="center">
<i>Training and validation loss curves showing triplet loss convergence and embedding distance separation across 2000 epochs. The twin/non-twin separation lines represent average embedding distances for twin pairs versus non-twin pairs across all pairs at each epoch. U-Net demonstrates most stable convergence with minimal overfitting and clear bimodal separation.</i>
</p>

## Connectome Workbench Clinical Integration

- **Medical format conversion**: Automated NIfTI and GIFTI format generation
- **Subject-specific atlas generation**: HCP-MMP 1.0 parcellation in native T1w space
- **GIFTI surface formats**: Cortical mapping for detailed hemisphere-specific visualization

![Connectome Workbench](assets/volumeandsurface.png)

<p align="center">
<i>Clinical visualization of genetic similarity importance maps in Connectome Workbench showing both volumetric (left) and surface-based (right) renderings overlayed on subject-specific T1w structural MRI. Gaussian-smoothed atlas heatmaps demonstrate enhanced spatial coherence in volume view and clear discrimination of high-importance regions in surface projection for clinical interpretation.</i>
</p>

## Installation

```bash
# Clone repository
git clone https://github.com/matthewkenely/3d-siamese-twin-ranking.git
cd 3d-siamese-twin-ranking

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
# Available at: https://drive.google.com/drive/folders/1AT22UDsgiR6NRpqpN0CBJRxVZWEFfX68?usp=sharing
```

## Citation

```bibtex
TBD
```
