# Fake News Detection with Transformer-based Cross-Domain Evaluation

## 📌 Project Overview
This project fine-tunes **DistilBERT** for fake news detection using the [FakeNewsNet dataset](https://github.com/KaiDMML/FakeNewsNet) containing **PolitiFact** and **GossipCop** news articles.  
The model is evaluated in both **in-domain** and **cross-domain** settings to analyze generalization performance across domains.

## 🚀 Features
- **Transformer-based Model**: Fine-tuned DistilBERT for binary classification (FAKE / REAL).
- **Balanced Training**: Applied class weighting to handle data imbalance.
- **Cross-Domain Evaluation**: Tested model trained on one domain against another to measure generalization.
- **Mitigation Strategies**: Experimented with:
  - Data augmentation (synonym replacement, random swaps)
  - Mixed-domain training (combining data from both domains)
  - Pseudo-labeling for unlabeled target domain data
- **Interactive Web App**: Built with **Streamlit** for real-time detection and SHAP-based interpretability.

## 📂 Dataset
**FakeNewsNet** with:
- **PolitiFact**: Political news articles
- **GossipCop**: Celebrity & entertainment news articles

| Domain       | Size    | FAKE % | REAL % |
|--------------|--------|--------|--------|
| PolitiFact   | 983    | 43.33% | 56.67% |
| GossipCop    | 20,743 | 23.02% | 76.98% |

## 📊 Results
| Experiment   | In-domain Accuracy | Cross-domain Accuracy | In-domain F1 | Cross-domain F1 |
|--------------|-------------------|-----------------------|--------------|-----------------|
| POL → GOSSIP | **83.76%**         | 36.47%                | 0.8644       | 0.3543          |
| GOSSIP → POL | **83.54%**         | 39.57%                | 0.8898       | 0.3501          |

## 🛠 Tech Stack
- **Language**: Python
- **Libraries**: PyTorch, Hugging Face Transformers, Datasets, Scikit-learn, Pandas, NumPy
- **Deployment**: Streamlit
- **Visualization**: SHAP, Matplotlib

## 📦 Installation
```bash
# Clone repository
git clone https://github.com/your-username/fake-news-detection-transformers.git
cd fake-news-detection-transformers

# Install dependencies
pip install -r requirements.txt
