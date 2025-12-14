# Credit Risk Probability Model using Alternative Data

## 📌 Project Overview

Bati Bank is partnering with an eCommerce platform to launch a **Buy-Now-Pay-Later (BNPL)** service.
This project develops a **credit risk probability model** using **alternative behavioral data** (transactions, RFM patterns) to assess customer creditworthiness in the absence of traditional credit bureau data.

The system:

* Engineers behavioral risk signals from transaction data
* Estimates **Probability of Default (PD)**
* Converts PD into a **credit score**
* Recommends **loan amount and duration**
* Exposes predictions via a **FastAPI service**



## 🎯 Business Objectives

* Enable BNPL lending for customers without formal credit history
* Minimize default risk while expanding financial inclusion
* Comply with **Basel II Capital Accord** risk management principles
* Build an auditable, scalable, and production-ready ML pipeline



## 🧠 Credit Scoring Business Understanding (Task 1)

### Basel II Accord and Risk Measurement

The Basel II Capital Accord requires banks to maintain sufficient capital based on quantified credit risk. This places strong emphasis on:

* **Probability of Default (PD) estimation**
* Transparency and auditability of models
* Robust documentation and governance

As a result, our credit scoring solution must be **interpretable, explainable, and well-documented**, enabling regulators and internal risk teams to understand how lending decisions are made and how risk is measured.



### Proxy Default Variable

The dataset does not include a direct loan default indicator. To overcome this, a **proxy default variable** is engineered using customer behavioral data derived from:

* **Recency**: How recently the customer transacted
* **Frequency**: How often the customer transacts
* **Monetary value**: How much the customer spends

Customers with low engagement, infrequent transactions, or declining spending behavior are treated as **higher risk**, while consistent and active users are treated as **lower risk**.

**Business Risks of Using a Proxy**

* Proxy may not perfectly represent true default behavior
* Potential misclassification of customers
* Risk of approving high-risk customers or rejecting low-risk ones

Despite these risks, proxy labeling is a common and accepted approach in alternative credit scoring when traditional labels are unavailable.



### Model Interpretability vs Performance Trade-off

In regulated financial environments, model choice involves trade-offs:

| Model Type                     | Pros                                          | Cons                                |
|  |  | -- |
| Logistic Regression (WoE)      | Highly interpretable, regulator-friendly      | Lower predictive power              |
| Tree-based / Gradient Boosting | Captures non-linear patterns, higher accuracy | Less transparent, harder to explain |

This project balances both by:

* Starting with interpretable baselines
* Using more complex models where justified
* Ensuring explainability and validation throughout



## 📊 Data Description

The dataset is sourced from an eCommerce transaction platform.

### Key Fields

* `TransactionId`: Unique transaction identifier
* `CustomerId`: Unique customer identifier
* `TransactionStartTime`: Timestamp of transaction
* `Amount`: Transaction amount (positive = debit, negative = credit)
* `Value`: Absolute transaction value
* `ProductCategory`, `ChannelId`, `ProviderId`
* `FraudResult`: Fraud flag (1 = fraud, 0 = not fraud)



## 🏗️ Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml     # CI/CD pipeline
├── data/
│   ├── raw/                    # Raw data (ignored)
│   └── processed/              # Processed features (ignored)
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
├── src/
│   ├── data_processing.py      # Feature engineering & proxy labels
│   ├── train.py                # Model training
│   ├── predict.py              # Inference logic
│   └── api/
│       ├── main.py             # FastAPI app
│       └── pydantic_models.py  # Request/response schemas
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```



## 🔬 Exploratory Data Analysis (Task 2)

EDA is performed in `notebooks/eda.ipynb` and covers:

* Data structure and types
* Missing value analysis
* Distribution of numerical and categorical features
* Outlier detection
* Correlation analysis
* Customer-level RFM behavior analysis



## ⚙️ Tech Stack

* **Python 3.10**
* **Pandas / NumPy** – data processing
* **Scikit-Learn** – modeling
* **FastAPI** – model serving
* **Docker** – containerization
* **GitHub Actions** – CI/CD
* **Pytest** – testing



## 🚀 Setup & Usage

### Local Setup

```bash
git clone https://github.com/<your-username>/credit-risk-model.git
cd credit-risk-model
pip install -r requirements.txt
```

### Run API with Docker

```bash
docker-compose up --build
```

API available at:

```
http://localhost:8000
```



## 🧪 Testing

```bash
pytest tests/
```

Tests run automatically via GitHub Actions on every push and pull request.







