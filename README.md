# causalCannibalisation

Work in progress

Repository hosting the code for the IEEE paper [Causal Quantification of Cannibalization during Promotional Sales in Grocery Retail](https://ieeexplore.ieee.org/document/9363114)


## Datasets
- Corporacion Favorita [Kaggle](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data)

- Breakfast at the Frat [Dunhumby](https://www.dunnhumby.com/source-files/)


## Libraries

- Papermill from [nteract](https://github.com/nteract/papermill), `pip install pycausalimpact`
- CausalImpact from [Dafiti](https://github.com/dafiti/causalimpact) `pip install pycausalimpact`




### Dunnhumby
From the csv data to the organised data (01.02.2021)
`Dunnhumby_arrange_store_sales.ipynb`

To summarise the sales per store (prior to the analysis)
`Dunnhumby_Summarise_store_sales.ipynb`

base NB to calculate the cannibalisation
`Dunnhumby_CausalImpact_Analysis_base.ipynb`


## Structure of the repo

The repo is structured as follows:

    .
    ├── README.md     <- This file ;)
    │
    ├── src
    │   │
    │   ├── notebooks   <- Collection of notebooks
    │   ├── notebooks/preprocessing_envelope_for_seasonality.ipynb <- STL preprocessing
    │   ├── notebooks/
    │   ├── notebooks/


### To create the graphs showed in the paper

The chart showing the STL decomposition of the total sales generated with `CFAV_store_sales_projection(paper).ipynb`

To summarise the sales per store (prior to the analysis) `CFAV_Summarise_store_sales.ipynb`

To summarise all the results `summarise_all_causal_results.ipynb`
For Dunnhumby data, use `Dunnhumby_summarise_all_causal_results.ipynb`

To run the surrogate model experiment, `Surrogate_model_experiment_paper.ipynb`

To explain how to select the promos `CFAV_show_selection_for_CausalImpact.ipynb`

To produce the **cannibalisation episode** plot `CFAV_CausalImpact_Analysis_Dairy_one_case(paper).ipynb`

To produce the **cannibalisation episode** using the Dunnhumby data, `Dunnhumby_CausalImpact_Analysis_Paper_plot.ipynb`

To produce the **graph** used in the paper `CFAV-causal_impact_GROCERY_I_Pichincha_49_A_11(graph-paper).ipynb`


## Installation as a Python wheel package
To generate the package from the source
```bash
python3 setup.py sdist bdist_wheel
```

In Python, just import as per
```python
from A.B import B
```