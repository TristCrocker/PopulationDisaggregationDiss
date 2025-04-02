
# Population Disaggregation using Graph Neural Networks in Sub-Saharan Africa

This repo is intended to be used as part of a 4th year computer science Dissertation at The University of Edinburgh.

This project is used for baseline result production and Graph Neural Network (GNN) result production in population disaggregation (admin level 2 to 3) on a Mozambican population and covariate dataset.

**Baseline Models**:
* Random Forest (RF)
* Bayesian Additive Regression Tree (BART)

**GNN Models**:
* GCNv2
* GATv2
* GraphSage
* Transformer GNN

## Getting Started (Baseline)

Cd into project and start R env,

```bash
cd dissBaseline/Bayesian-Top-Down-Modelling
R
```

To run RF baseline train/test,

```bash
source("RF_Baseline_Workflow.R")
```

To run BART baseline train/test,

```bash
source("BART_Baseline_Workflow.R")

```

## Getting Started (GNN)

Cd into project,

```bash
cd dissGNN
```

Intall dependencies,

```bash
pip install -r requirements.txt
```

To run GNN models train/test,
```bash
python -m main
```

(Some code is commented out for things like hyperparam search and visualisation, uncomment to see these)

## Languages Used 

* Python (GNNs)
* R (Baselines)

## Acknowledgements 

* The baseline scripts were adapted from https://github.com/wpgp/Bayesian-Top-Down-Modelling
* The Mozambican dataset was provided by Beate Desmitniece and Sean Ó Héir.




