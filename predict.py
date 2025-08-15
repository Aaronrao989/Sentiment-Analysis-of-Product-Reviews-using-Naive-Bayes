#!/usr/bin/env python3
"""
Amazon Reviews Sentiment Analysis - Prediction Script
This script loads a trained Naive Bayes model and makes predictions on new text.
"""

import pickle
import re
import string
import math
from collections import Counter
import argparse

import nltk
from nltk.stem import SnowballStemmer 
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize

# Initialize preprocessing tools
try:
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
except:
    print("Warning: NLTK data not found. Run 'python -m nltk.downloader punkt stopwords' first.")
    stemmer = None
    stop_words = set()

def preprocess_text(text):
    """
    Preprocess text data for sentiment analysis
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Cleaned and preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    if stemmer:
        words = word_tokenize(text)
        # Remove stopwords and stem
        words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
    else:
        # Fallback if NLTK is not properly configured
        words = [w for w in text.split() if len(w) > 2]
    
    return ' '.join(words)

def make_class_prediction(text, counts, class_prob, class_count):
    """Make prediction for a single class using log probabilities"""
    log_prediction = math.log(class_prob)
    text_counts = Counter(re.split(r"\s+", text))
    total_words = sum(counts.values()) + class_count
    
    for word in text_counts:
        word_prob = (counts.get(word, 0) + 1) / total_words
        log_prediction += text_counts[word] * math.log(word_prob)
    
    return log_prediction

class SentimentAnalyzer:
    """Sentiment Analysis class for Amazon reviews"""
    
    def __init__(self, model_path='models/naive_bayes_model.pkl'):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model = None
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded from {self.model_path}")
        except FileNotFoundError:
            print(f"❌ Model file not found at {self.model_path}")
            print("Please train the model first using train_model.py")
            self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def predict(self, text, return_confidence=False):
        """
        Predict sentiment for a given text
        
        Args:
            text (str): Input text to analyze
            return_confidence (bool): Whether to return confidence scores
            
        Returns:
            str or tuple: Sentiment prediction ('POSITIVE' or 'NEGATIVE')
                         If return_confidence=True, returns (prediction, pos_conf, neg_conf)
        """
        if not self.model:
            return "ERROR: Model not loaded"
        
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Make predictions
        neg_pred = make_class_prediction(
            processed_text, 
            self.model['negative_counts'], 
            self.model['prob_negative'], 
            self.model['negative_review_count']
        )
        
        pos_pred = make_class_prediction(
            processed_text, 
            self.model['positive_counts'], 
            self.model['prob_positive'], 
            self.model['positive_review_count']
        )
        
        # Determine prediction
        prediction = "POSITIVE" if pos_pred > neg_pred else "NEGATIVE"
        
        if return_confidence:
            # Convert log probabilities to confidence scores (0-1)
            # Higher score = more confident
            pos_confidence = math.exp(pos_pred - max(pos_pred, neg_pred))
            neg_confidence = math.exp(neg_pred - max(pos_pred, neg_pred))
            
            # Normalize
            total = pos_confidence + neg_confidence
            pos_confidence /= total
            neg_confidence /= total
            
            return prediction, pos_confidence, neg_confidence
        
        return prediction
    
    def predict_batch(self, texts):
        """
        Predict sentiment for a list of texts
        
        Args:
            texts (list): List of text strings to analyze
            
        Returns:
            list: List of predictions
        """
        return [self.predict(text) for text in texts]

def interactive_mode():
    """Run interactive prediction mode"""
    analyzer = SentimentAnalyzer()
    
    if not analyzer.model:
        return
    
    print("\n🔮 Interactive Sentiment Analysis")
    print("=" * 40)
    print("Enter reviews to analyze sentiment (type 'quit' to exit)")
    print("For confidence scores, add '--conf' after your review")
    
    while True:
        try:
            user_input = input("\n📝 Enter review: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Check if user wants confidence scores
            show_confidence = False
            if user_input.endswith('--conf'):
                show_confidence = True
                user_input = user_input.replace('--conf', '').strip()
            
            # Make prediction
            if show_confidence:
                prediction, pos_conf, neg_conf = analyzer.predict(user_input, return_confidence=True)
                print(f"🎯 Sentiment: {prediction}")
                print(f"📊 Confidence - Positive: {pos_conf:.1%}, Negative: {neg_conf:.1%}")
            else:
                prediction = analyzer.predict(user_input)
                print(f"🎯 Sentiment: {prediction}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_mode(input_file, output_file=None):
    """
    Run batch prediction mode
    
    Args:
        input_file (str): Path to input file (one review per line)
        output_file (str): Path to output file (optional)
    """
    analyzer = SentimentAnalyzer()
    
    if not analyzer.model:
        return
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reviews = [line.strip() for line in f if line.strip()]
        
        print(f"📂 Processing {len(reviews)} reviews from {input_file}")
        
        predictions = []
        for i, review in enumerate(reviews, 1):
            pred = analyzer.predict(review)
            predictions.append((review, pred))
            if i % 100 == 0:
                print(f"✓ Processed {i}/{len(reviews)} reviews")
        
        # Save results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Review\tPrediction\n")
                for review, pred in predictions:
                    f.write(f"{review}\t{pred}\n")
            print(f"💾 Results saved to {output_file}")
        else:
            print("\n📊 Results:")
            print("-" * 80)
            for review, pred in predictions:
                print(f"{pred}: {review[:60]}{'...' if len(review) > 60 else ''}")
        
        # Summary
        pos_count = sum(1 for _, pred in predictions if pred == 'POSITIVE')
        neg_count = len(predictions) - pos_count
        print(f"\n📈 Summary: {pos_count} Positive, {neg_count} Negative")
        
    except FileNotFoundError:
        print(f"❌ Input file not found: {input_file}")
    except Exception as e:
        print(f"❌ Error processing batch: {e}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Amazon Reviews Sentiment Analysis - Prediction')
    parser.add_argument('--mode', choices=['interactive', 'batch'], default='interactive',
                       help='Prediction mode (default: interactive)')
    parser.add_argument('--input', '-i', help='Input file for batch mode')
    parser.add_argument('--output', '-o', help='Output file for batch mode')
    parser.add_argument('--model', '-m', default='models/naive_bayes_model.pkl',
                       help='Path to model file')
    parser.add_argument('--text', '-t', help='Single text to analyze')
    
    args = parser.parse_args()
    
    print("🎯 Amazon Reviews Sentiment Analysis - Prediction")
    print("=" * 55)
    
    if args.text:
        # Single prediction mode
        analyzer = SentimentAnalyzer(args.model)
        if analyzer.model:
            prediction = analyzer.predict(args.text)
            print(f"Text: {args.text}")
            print(f"Sentiment: {prediction}")
    elif args.mode == 'batch':
        if not args.input:
            print("❌ Input file required for batch mode. Use --input flag.")
            return
        batch_mode(args.input, args.output)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
