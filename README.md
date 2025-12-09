# 🧠 ReviewPulse – Live Sentiment predictions using deep learning models (BERT, LSTM, GRU, and RNN)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)

This project performs sentiment analysis using four deep learning models — BERT, LSTM, GRU, and Simple RNN — and systematically compares them in terms of classification accuracy, F1-score, computational efficiency, and implementation complexity. The pipeline was built using PyTorch and Hugging Face Transformers, incorporating custom preprocessing for Word2Vec/GloVe-style models (vocabulary indexing, sequence padding) and tokenizer-based preprocessing for BERT.

Multiple training strategies were evaluated, including learning rate optimization, early stopping, checkpoint-based model saving, and weighted loss handling for class imbalance, enabling reproducible and stable training. The results highlight the performance advantages of contextual embeddings, with BERT outperforming classical RNN-based architectures by a significant margin. 

The project concludes with the development of a Real-Time Sentiment Prediction application capable of loading trained weight files and performing on-the-fly preprocessing and inference, providing users with comparative confidence scores across all four models for unseen inputs. This enables interactive evaluation of model behavior beyond offline metrics and emphasizes practical deployment considerations.


---

## 📁 Project Structure
```
NLP-Sentiment-Model-Comparison/
├── main.ipynb 		# Main notebook with training and evaluation
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project overview


```
## Dataset and Models

**Models** :  https://drive.google.com/drive/folders/1KyBVUQT_m9mhYF8p5eI3m7wFqKZSWRE3?usp=sharing
**Dataset-BERT** : https://www.kaggle.com/datasets/virajjayant/bertbaseuncased
**Dataset-Glove** : https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt
**Dataset-IMDB Movie review** : https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews


---

## 🧠 Models Compared

|   Model  |  			   Summary 			    |
|----------|--------------------------------------------------------|
| **BERT** | Pre-trained transformer from Hugging Face (fine-tuned) |
| **LSTM** | Long Short-Term Memory network for sequential modeling |
| **GRU**  | Gated Recurrent Unit for efficient RNN-based modeling |
| **RNN**  | Baseline simple Recurrent Neural Network |

---

📊 Evaluation Metrics

✅ Accuracy
✅ Precision, Recall, F1-Score
✅ Confusion Matrix
✅ ROC-AUC
✅ Training Time & Memory Usage

---

🛠 Built With

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- scikit-learn
- Jupyter Notebook
- matplotlib, seaborn, numpy, pandas

---
