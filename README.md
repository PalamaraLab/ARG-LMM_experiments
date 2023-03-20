## Simulation and analysis scripts for Zhang et al. Nature Genetics, 2023

This repository contains scripts used for most analyses described in [Zhang et al.](https://www.nature.com/articles/s41588-023-01379-x), Biobank-scale inference of ancestral recombination graphs enables genealogical analysis of complex traits, Nature Genetics, 2023 (see also the [Zenodo repository](https://zenodo.org/records/7745746)). These include ARG-based linear mixed model (ARG-LMM) analysis and simulation of complex traits, genealogy-wide association analysis in the UK Biobank, computation of ARG accuracy metrics, and genotype imputation.

### Setup
You will want to install Python packages according to the requirements.txt file. These include `arg-needle`, `arg-needle-lib`, `msprime`, `tskit`, `tsinfer`, and `tszip`. At the root of this directory, you should also create symlinks to the following binaries as needed:
```
beagle5.jar
bolt
gcta
impute4
plink2
Relate
RelateFileFormats
```

### Directory structure
There are three experiment-related folders -- `metrics/`, `arg_lmm/`, and `impute/` -- as well as one folder with common utilities and setup (`common/`). The desired experiment can be run using `python3 path/to/file.py` -- the directory the file is run from should not matter. Roughly, `metrics/` is used for benchmarking ARG inference algorithms, `arg_lmm/` is used for ARG-based linear mixed model (ARG-LMM) analyses, including the calculation of ARG-GRMs and their use in heritability, prediction, and association analyses (e.g. ARG-MLMA) in simulations using true or ARG-Needle inferred ARGs, and `impute/` is used for ARG-based imputation. This code is meant to support reproducing the analyses described in the paper. Note that some APIs may have changed during development and some scripts may not work as expected, but should still help in reproducing the analyses. For up-to-date scripts and workflows for several analyses described here, please refer to [this page](https://palamaralab.github.io/software/argneedle/).

### Real data experiments
The folder `real/` contains example scripts which are referenced in the ARG-Needle inference and association [manual](https://palamaralab.github.io/software/argneedle/). These scripts can be used to recreate the ARG-Needle and ARG-MLMA steps with UK Biobank data obtained separately (see [manual](https://palamaralab.github.io/software/argneedle/) and paper for additional details). We have made COJO association statistics and filtered trait association results available elsewhere (see paper).

### License
This code is distributed under the GNU General Public License v3.0 (GPLv3). If you use this software, please cite: [Zhang et al.](https://www.nature.com/articles/s41588-023-01379-x), Biobank-scale inference of ancestral recombination graphs enables genealogical analysis of complex traits, Nature Genetics, 2023.
