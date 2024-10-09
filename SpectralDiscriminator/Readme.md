# Run the following command to get an expert model that estimates the maximum emission wavelength.

python graph_main7.py --train_csv='emission_sol.csv' --test_csv='chromophore_novo.csv'

### The code we use is adapted from the following article. See the original model article for more details.

```
Guo, J.; Sun, M.; Zhao, X.; Shi, C.; Su, H.; Guo, Y.; Pu, X. General Graph Neural Network-Based Model To Accurately Predict Cocrystal Density and Insight from Data Quality and Feature Representation. J. Chem. Inf. Model. 2023, 63 (4), 1143–1156.
```