# AML Financial Crime Analytics

End-to-end portfolio demonstration using the Kaggle Synthetic Transaction Monitoring Dataset (SAML-D).

## Questions answered
1. What fraction of transactions have laundering labels?
2. Which transaction types have higher observed label rates?
3. What hours have higher observed label rates?
4. How does account activity relate to at-least-one positive label?
5. What account-network structures warrant investigation?
6. How well do Random Forest and XGBoost identify labelled transactions?
7. What alert workload results from the validation-selected threshold?

Actual numerical answers are generated in reports/business/business_answers.json; do not invent them.

## Setup (Windows PowerShell, PyCharm)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Configure Kaggle API credentials according to Kaggle's instructions, then run:
python src/download_data.py
python src/prepare_data.py
python src/train_models.py
python src/evaluate_models.py
python src/network_analysis.py
python src/explain_models.py
python src/business_insights.py
python -m streamlit run dashboard/app.py 