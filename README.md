# Data Security and Privacy in the AI and Post-Quantum Era

### Emerging Threats, Challenges, and Countermeasures

A Python-based research project investigating data security and privacy risks associated with Artificial Intelligence (AI), organizational security practices, and the adoption of Post-Quantum Cryptography (PQC).

---

## 📌 About the Project

As Artificial Intelligence becomes increasingly integrated into organizational systems, protecting sensitive information and ensuring secure data processing have become important cybersecurity challenges. At the same time, advances in quantum computing may threaten some of the cryptographic techniques currently used to protect digital information.

This project analyzes synthetic organizational security data to explore the relationships between AI usage, data sensitivity, privacy risks, AI security risks, security controls, legacy system dependencies, and Post-Quantum Cryptography (PQC) adoption.

The project combines statistical analysis, data visualization, correlation analysis, and machine learning to examine these relationships and support research into emerging cybersecurity threats and potential countermeasures.

## 🎯 Objectives

- Analyze the relationship between AI usage and AI security risk.
- Investigate how data sensitivity relates to privacy risk.
- Examine the relationship between access control strength and incident probability.
- Explore the potential significance of legacy system dependencies in organizational security.
- Study security-related patterns across different PQC adoption levels.
- Apply machine learning models to assess risk-related patterns in the dataset.
- Generate visualizations and statistical summaries to support research findings.

## 🛠️ Technologies Used

- **Python** — Primary programming language.
- **Pandas** — Data manipulation and analysis.
- **NumPy** — Numerical operations.
- **Matplotlib** — Data visualization.
- **Seaborn** — Statistical visualizations.
- **Scikit-learn** — Data preprocessing, model training, and evaluation.

## 📊 Research Methodology

The analysis workflow includes:

1. **Data preparation:** Load the synthetic dataset, identify missing values, and prepare variables for analysis.
2. **Exploratory data analysis:** Examine distributions and summarize important security-related variables.
3. **Correlation analysis:** Investigate associations between relevant variables.
4. **Data visualization:** Generate colorful charts, correlation heatmaps, and comparative plots.
5. **Machine learning:** Evaluate Logistic Regression and Random Forest models using appropriate classification metrics.
6. **Model evaluation:** Review performance metrics and visualizations, including confusion matrices and ROC curves where applicable.
7. **Interpretation:** Discuss the results in the context of the synthetic dataset and the research objectives.

## 🤖 Machine Learning Models

The project uses the following classification algorithms:

- **Logistic Regression:** A baseline model for classification and risk-related pattern analysis.
- **Random Forest:** An ensemble learning model used to examine classification performance and feature importance.

Model performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC when applicable to the classification task.

Actual results depend on the dataset, preprocessing steps, target variable, and experimental configuration.

## 📈 Key Areas of Analysis

The project focuses on the following research questions:

- Is higher AI usage associated with higher AI security risk?
- Does greater data sensitivity correspond to higher privacy risk?
- How is access control strength associated with incident probability?
- What security patterns appear across different levels of PQC adoption?
- Which input features contribute most to the evaluated machine learning models?

These questions are investigated through statistical summaries, visual comparisons, and model-based analysis. The observed relationships should not be interpreted as proof of causation.

## 📁 Project Structure

```text
ai-post-quantum-data-security-analysis/
│
├── AI_Post_Quantum_Research_Analysis.py
├── README.md
├── .gitignore
│
├── Research_Paper_Figures/
│   ├── Visualizations (.png)
│   └── Analysis Outputs (.csv)
│
└── Dataset/
    └── Synthetic Dataset (.csv)
```

*Note: The folder structure above is illustrative. Your actual repository may differ depending on which files you upload. Generated figures, CSV outputs, and the dataset should be included only if you choose to publish them.*

## ⚙️ Installation and Setup

### Prerequisites

Install Python 3.10 or a compatible version supported by the selected library releases.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/ai-post-quantum-data-security-analysis.git
cd ai-post-quantum-data-security-analysis
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Configure the Dataset Path

Open `AI_Post_Quantum_Research_Analysis.py` and update the dataset path to the location of your CSV file.

Ensure the dataset contains the columns expected by the script.

### 4. Run the Analysis

```bash
python AI_Post_Quantum_Research_Analysis.py
```

The script generates analysis outputs and visualizations in the configured output directory, provided that the dataset and dependencies are available and the execution completes successfully.

## 🔐 Research Scope and Limitations

- The project uses synthetic organizational security data rather than verified records from real organizations.
- Findings describe patterns within the analyzed dataset and may not generalize to real-world environments.
- Correlation does not establish causation.
- Machine learning performance depends on dataset construction, feature selection, preprocessing, and the evaluation procedure.
- PQC adoption categories alone do not establish that an organization is secure or resistant to quantum attacks.

The results should therefore be interpreted as exploratory research findings, not as definitive evidence of real-world security outcomes.

## 🔮 Future Enhancements

Potential future improvements include:

- Evaluating the analysis on validated real-world or benchmark datasets.
- Expanding the comparison of machine learning algorithms.
- Improving model validation and robustness testing.
- Investigating privacy-preserving machine learning approaches.
- Examining NIST-standardized post-quantum cryptographic algorithms and cryptographic agility.
- Developing an interactive dashboard for exploring security patterns.

## 📚 Research Context

This project is associated with research on AI security, privacy-preserving machine learning, data extraction and inference attacks, and post-quantum cryptography.

Relevant background resources include publications by Shokri et al. on membership inference, Abadi et al. on differential privacy, Tramèr et al. on model extraction, Carlini et al. on training-data extraction, and guidance published by NIST on post-quantum cryptography and AI risk management.

Consult the original publications and standards for complete bibliographic details.

## 👩‍💻 Author

**Radhika Gupta**  
Department of Information Technology  
Ramniranjan Jhunjhunwala College

**Research Topic:** Data Security and Privacy in the AI and Post-Quantum Era: Emerging Threats, Challenges, and Countermeasures.

---

*This repository contains research analysis code intended for educational and academic purposes.*
