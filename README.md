# Drug Design for Diabetes (Team 29)

This repository contains the work of Team 29 for the Master's course project on Master degree on Applied Artificial Intelligence .  
We aim to design therapeutic candidates for **diabetes**, combining generative protein design methods with machine learning predictors to evaluate their properties.

---

Our workflow follows an iterative pipeline:
1. **Backbone generation**  
   - Using [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) to generate protein backbones.

2. **Sequence design**  
   - Applying [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) to generate amino acid sequences compatible with the backbone.

3. **Predictive evaluation**  
   - Machine learning models are used to assess the designed proteins in terms of:
     - Binding affinity
     - Stability
     - Solubility
     - Toxicity  

   We apply Support Vector Machines, Random Forest, and XGBoost predictors.  

4. **Iteration**  
   - Promising candidates are kept for further analysis.  
   - Failed candidates (low score in predictors) are discarded, and the generation cycle is repeated.


## 📂 Repository Structure

drug-design-diabetes-team29/
│── notebooks/
│ └── Avance1.#29.ipynb 
│── README.md # Project documentation
│── requirements.txt # Dependencies for notebooks
