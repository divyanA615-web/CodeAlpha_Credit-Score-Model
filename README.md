# 🏆 Credit Score Classification Model
### CodeAlpha Machine Learning Internship — Task 3

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/CatBoost-1.2+-FF6B35?style=for-the-badge&logo=yandex&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Completed-2ecc71?style=for-the-badge"/>
</p>

<p align="center">
  <b>Multi-class credit score classification using real-world financial data.</b><br/>
  Predicts whether a person's credit score is <b>Poor</b>, <b>Standard</b>, or <b>Good</b>
  based on 28 financial and behavioural features.
</p>

---

## 📌 Problem Statement

> Financial institutions rely on credit scores to evaluate a customer's creditworthiness before approving loans, credit cards, or mortgages. Manual evaluation is slow, inconsistent, and prone to human bias.

This project builds an **automated ML pipeline** that classifies credit scores into three categories:

| Class | Label | Meaning |
|-------|-------|---------|
| `0` | 🔴 Poor | High risk — likely to default |
| `1` | 🟡 Standard | Moderate risk — average creditworthiness |
| `2` | 🟢 Good | Low risk — financially responsible |

---

## 📂 Repository Structure

```
CodeAlpha_Credit-Score-Model/
│
├── 📄 credit_score.py          # Main ML pipeline (cleaning → training → evaluation)
├── 📄 scratch_inspect.py       # Data inspection & exploratory analysis script
├── 📊 train_data.csv           # Raw training dataset (28 features, ~100K rows)
│
├── 📁 catboost_info/           # CatBoost training logs & metadata (auto-generated)
├── 📁 train_output/            # Model outputs, charts & submission files
│
├── 📄 .gitignore
└── 📄 README.md
```

---

## 📊 Dataset Overview

| Property | Detail |
|----------|--------|
| **Rows** | ~100,000 training samples |
| **Features** | 28 raw input columns |
| **Target** | `credit_score` (0 = Poor, 1 = Standard, 2 = Good) |
| **Source** | CodeAlpha provided dataset |
| **Format** | CSV (ZIP-compressed) |

### Key Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `annual_income` | Float | Customer's yearly income |
| `outstanding_debt` | Float | Total unpaid debt |
| `credit_utilization_ratio` | Float | % of credit limit used |
| `num_of_delayed_payment` | Integer | Number of late payments |
| `credit_history_age` | String → Int | Length of credit history (months) |
| `monthly_balance` | Float | Average monthly remaining balance |
| `total_emi_per_month` | Float | Monthly loan instalment total |
| `payment_behaviour` | Categorical | Spending and payment pattern |
| `credit_mix` | Categorical | Mix of credit types (Good/Standard/Bad) |
| `interest_rate` | Integer | Average interest rate on credit |
| `num_of_loan` | Integer | Number of active loans |
| `delay_from_due_date` | Integer | Average days delayed on payments |

---

## 🛠️ Tech Stack

```
Language     : Python 3.11+
ML Model     : CatBoost (Gradient Boosting)
Data         : Pandas, NumPy
Evaluation   : Scikit-learn (F1, Accuracy, Confusion Matrix)
Visualization: Matplotlib, Seaborn
```

---

## ⚙️ Pipeline Architecture

```
Raw CSV
   │
   ▼
┌─────────────────────────────────────────┐
│  STEP 1 — Data Loading                 │
│  • ZIP-safe CSV reading                │
│  • Shape & dtype inspection            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  STEP 2 — Data Cleaning                │
│  • Sentinel string removal (___, !@#)  │
│  • Trailing character stripping        │
│  • NLP parsing: "22 Years 5 Months"   │
│  • Out-of-range value clipping         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  STEP 3 — Feature Engineering          │
│  • Debt-to-Income ratio                │
│  • EMI Burden ratio                    │
│  • Credit Age (months → numeric)       │
│  • Loan type binary flags              │
│  • Payment behaviour risk score        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  STEP 4 — Encoding & Preprocessing     │
│  • Label Encoding (categoricals)       │
│  • Median imputation (NaN fill)        │
│  • IQR-based outlier clipping          │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  STEP 5 — Model Training               │
│  • CatBoostClassifier                  │
│  • StratifiedKFold Cross-Validation    │
│  • Early stopping on validation loss   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  STEP 6 — Evaluation & Output          │
│  • Weighted F1 Score                   │
│  • Confusion Matrix                    │
│  • Per-class Precision / Recall        │
│  • submission.csv generation           │
└─────────────────────────────────────────┘
```

---

## 🧹 Key Data Cleaning Challenges

This dataset contained several real-world data quality issues that required custom solutions:

| Problem | Example | Solution |
|---------|---------|----------|
| Sentinel garbage strings | `"_______"`, `"!@9#%8"` | Mapped to `NaN` before casting |
| Trailing special characters | `"3_"`, `"25!"` | Regex strip → numeric cast |
| Natural language fields | `"22 Years and 5 Months"` | Custom regex parser → integer months |
| ZIP-compressed CSV | File appears as `.csv` but is binary | Detected format, used `compression='zip'` |
| Out-of-range values | Age = 500, Interest = 1000% | Domain-knowledge based clipping |

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Weighted F1 Score** | ~0.78 |
| **Train Accuracy** | ~88% |
| **Validation Strategy** | 3-Fold Stratified KFold |
| **Model** | CatBoostClassifier |

> ⚠️ Results are on a cross-validated OOF (Out-of-Fold) basis to prevent data leakage.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/divyanA615-web/CodeAlpha_Credit-Score-Model.git
cd CodeAlpha_Credit-Score-Model
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn catboost matplotlib seaborn
```

### 3. Run the pipeline
```bash
python credit_score.py
```

### 4. Inspect the data (optional)
```bash
python scratch_inspect.py
```

---

## 📦 Dependencies

```txt
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
catboost>=1.2
matplotlib>=3.8
seaborn>=0.13
```

---

## 💡 Feature Engineering Highlights

Beyond raw features, the following **engineered signals** were derived:

```python
debt_to_income       = outstanding_debt / annual_income
emi_burden_ratio     = total_emi_per_month / monthly_inhand_salary
savings_rate         = amount_invested_monthly / monthly_inhand_salary
delay_severity       = delay_from_due_date × num_of_delayed_payment
credit_age_years     = credit_history_age / 12
card_per_account     = num_credit_card / (num_bank_accounts + 1)
outstanding_debt_log = log1p(outstanding_debt)     # handle skew
```

These ratio-based features significantly boosted model performance by capturing **financial stress patterns** invisible in the raw columns.

---

## 🎯 Business Impact

This model can help banks and financial institutions:

- ✅ **Automate** credit score decisions in milliseconds
- ✅ **Reduce bias** from manual human review
- ✅ **Identify high-risk** customers before loan approval
- ✅ **Segment** customers for targeted financial products
- ✅ **Save costs** on manual credit bureau checks

---

## 🧠 Lessons Learned

1. **Real-world data is messy** — sentinel strings and encoding issues need custom parsers
2. **File format detection matters** — ZIP-compressed CSVs silently break `pd.read_csv()`
3. **Ratio features add signal** — debt-to-income, EMI burden outperformed raw columns
4. **Stratified CV is essential** — class imbalance makes random splits unreliable
5. **CatBoost handles categoricals natively** — no need for manual one-hot encoding

---

## 👤 Author

**Divyan A615**
- 🎓 CodeAlpha Machine Learning Internship
- 🐙 GitHub: [@divyanA615-web](https://github.com/divyanA615-web)

---

## 📜 License

This project is built as part of the **CodeAlpha ML Internship Program**.  
Feel free to fork, star ⭐, and use for educational purposes.

---

<p align="center">
  Made with ❤️ by <b>Divyan</b> | CodeAlpha ML Intern
</p>
