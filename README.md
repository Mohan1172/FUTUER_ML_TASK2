# Customer Support Ticket Classification and Priority System

A beginner-friendly Python/NLP project that predicts a support ticket's **category** and assigns a **high / medium / low priority**. It uses scikit-learn TF-IDF + Logistic Regression for category classification and transparent rules for priority tagging.

## Features
- Text normalization and cleaning
- TF-IDF text vectorization and machine-learning category prediction
- Rule-based priority tagging with explainable reasons
- Evaluation with accuracy, classification report, and confusion matrix
- Command-line prediction for a new ticket
- Small, included sample dataset so the project runs without downloading data
- Jupyter Notebook for step-by-step exploration

## Project structure
```text
Customer_Support_Ticket_Classifier/
├── app.py                 # Click Run to launch the local website
├── data/
│   └── support_tickets.csv
├── notebooks/
│   └── ticket_classification_walkthrough.ipynb
├── src/
│   └── ticket_classifier.py
├── tests/
│   └── test_ticket_classifier.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 1. Requirements
- Python 3.10 or newer (Python 3.11 or 3.12 is a good choice)
- Internet is needed only to install Python packages. The included dataset is local.

## 2. Run the website from VS Code (recommended)

After installing the requirements, open `app.py` in VS Code and click **Run Python File** (the play button). A local website should open in your browser. If it does not, open `http://127.0.0.1:5000` manually. Keep the VS Code terminal running while using the website; press `Ctrl+C` in the terminal to stop it.

The first-time setup still requires installing Python packages as shown below.

## 3. Run the command-line demo

Open this folder in VS Code, then open Terminal → New Terminal.

### Windows
```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/ticket_classifier.py
```

If `py` is unavailable, use `python` instead. If PowerShell blocks activation, you can skip activation and run `.venv\Scripts\python.exe -m pip install -r requirements.txt`, then `.venv\Scripts\python.exe src/ticket_classifier.py`.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/ticket_classifier.py
```

The script trains the model, prints evaluation metrics, and demonstrates predictions. No API key, cloud service, database, or downloaded model is required.

## 4. Predict your own ticket
After installing requirements, run:
```bash
python src/ticket_classifier.py --ticket "I was charged twice and need a refund immediately"
```
Optional category labels include `billing`, `technical`, `account`, `shipping`, and `cancellation_refund`. Priority is assigned by the rule set described below.

## 5. Jupyter Notebook
Install the requirements, then run:
```bash
jupyter notebook
```
Open `notebooks/ticket_classification_walkthrough.ipynb` and run the cells from top to bottom. In VS Code, you can also open the notebook directly if the Jupyter extension is installed.

## 6. How it works
1. **Text cleaning:** lowercases text, removes URLs and punctuation, and collapses extra spaces. Scikit-learn's TF-IDF vectorizer handles tokenization and stop-word filtering.
2. **Category model:** a TF-IDF + Logistic Regression pipeline learns from the labeled ticket examples.
3. **Priority logic:** transparent rules identify urgent wording and high-impact situations. High priority is assigned for explicit urgent/safety/outage signals; medium for moderate urgency or payment/refund/account-access issues; otherwise low. Review and adjust the rules for your organization's policies.
4. **Evaluation:** a stratified train/test split and standard classification metrics estimate category-model performance.

## Dataset and limitations
`data/support_tickets.csv` is a small **illustrative sample dataset**, created for learning and demonstration. It is not real customer data and its metrics should not be treated as production evidence. A model trained on a larger, representative, reviewed dataset will usually generalize better. The priority rules are a baseline, not a replacement for human judgment. Do not put real personal, financial, or sensitive customer data in a public repository.

## GitHub upload
1. Create a public repository named `Customer_Support_Ticket_Classifier`.
2. Upload the contents of this folder (upload the files inside it, not the ZIP itself, if you want GitHub to display the project files).
3. Commit `README.md`, `requirements.txt`, `data/`, `src/`, `notebooks/`, and `tests/`.
4. Anyone can then clone/download the repository and follow the run steps above. They must install Python packages first.

## License
MIT — you may use, modify, and share this learning project.
