# **Toxic Comment Classification using Word2Vec and Deep Learning Models**

## **📌 Overview**
This project implements **toxic comment classification** using different deep learning models with **Word2Vec embeddings**. It includes:
- **Pure RNN (GRU) with Word2Vec** (Pretrained & Non-pretrained)
- **Pure CNN with Word2Vec**
- **Hybrid CNN + RNN (Bi-GRU) with Word2Vec**

## **🚀 Requirements**
Before running the code, install the required libraries:
```bash
pip install numpy pandas tensorflow gensim
```

## **🔹 How to Use**
1. **Download the Dataset:**  
   - Place `train.csv` and `test.csv` inside `./input/` folder.
   - Ensure `sample_submission.csv` is also available for submission generation.

2. **Download Pretrained Word2Vec (Optional for Pretrained Models)**  
   - Download **GoogleNews-vectors-negative300.bin** from [Google Drive](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz)
   - Extract it using:
     ```bash
     gunzip GoogleNews-vectors-negative300.bin.gz
     ```
   - Place it inside the `./input/` folder.

3. **Run the Code**  
   - Open and run `code.ipynb` in Jupyter Notebook or execute:
     ```bash
     python code.py
     ```
   - The script automatically **trains Word2Vec from scratch** if not using pretrained embeddings.

4. **Evaluate Model Performance**  
   - The script prints **accuracy, loss, and validation performance** for each model.
   - The **best model is saved** as `weights_best.keras`.

5. **Make Predictions & Submit**  
   - The script generates predictions and saves them as `baseline_word2vec_rnn.csv`.

## **🔹 Model Comparisons**
| **Model** | **Embedding Type** | **Validation Accuracy** | **Training Speed** |
|-----------|----------------|--------------------|----------------|
| Pretrained Word2Vec + GRU | Frozen Word2Vec (300D) | ✅ High (98.8%) | ✅ Fast |
| Non-pretrained Word2Vec + GRU | Trained Word2Vec (100D) | ✅ Very High (99.3%) | ❌ Slow |
| Pure CNN + Word2Vec | Trained Word2Vec (100D) | ✅ Good (99.1%) | ⏳ Medium |
| Hybrid CNN + RNN (Bi-GRU) | Trained Word2Vec (100D) | 🚀 Best (99.3%) | ⏳ Medium |
