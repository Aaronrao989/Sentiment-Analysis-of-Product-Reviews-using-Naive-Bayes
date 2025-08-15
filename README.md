# Amazon Reviews Sentiment Analysis

A simple sentiment analysis tool that classifies Amazon product reviews as **positive** or **negative** using Naive Bayes.

## 🚀 Quick Start

1. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn nltk
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

2. **Train the model**
   ```bash
   python train_model.py
   ```

3. **Make predictions**
   ```bash
   python predict.py --text "This product is amazing!"
   ```

## 📁 Files

- `train_model.py` - Train the sentiment analysis model
- `predict.py` - Make predictions on new text
- `amazon-reviews-sentiment-analysis.ipynb` - Jupyter notebook with full analysis
- `requirements.txt` - Python dependencies

## 🎯 Usage Examples

**Single prediction:**
```bash
python predict.py --text "Great product, love it!"
# Output: POSITIVE
```

**Interactive mode:**
```bash
python predict.py
# Enter reviews one by one
```

**Batch processing:**
```bash
python predict.py --mode batch --input reviews.txt --output results.txt
```

## 📊 Performance

- **Accuracy**: 78%
- **Dataset**: 34K+ Amazon reviews
- **Classes**: Positive (≥4 stars) vs Negative (<4 stars)

## 🛠️ How it works

1. **Preprocess text** - Remove punctuation, stopwords, apply stemming
2. **Balance data** - Equal positive/negative samples
3. **Train Naive Bayes** - Multinomial classifier with Laplace smoothing
4. **Evaluate** - Classification report, confusion matrix, AUC

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
nltk>=3.6.0
```

## 🤔 Need help?

- Make sure your dataset is in `data/` folder
- Run training before prediction
- Check that NLTK data is downloaded

---

**⭐ Star this repo if it helped you!**
