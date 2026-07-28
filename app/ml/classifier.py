"""
classifier.py - TensorFlow text classification model for document categorization.

Pipeline:
  1. Data preprocessing & feature engineering (TF-IDF style tokenization)
  2. Model architecture (Embedding → GlobalAveragePooling → Dense)
  3. Training with synthetic labeled data
  4. Model evaluation
  5. Model persistence (.keras + tokenizer.json)
  6. Prediction API
"""

import os
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Optional

from app.core.config import settings
from app.core.logging import logger

# ── Lazy TensorFlow import to avoid slow startup ──────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed — classification will be unavailable.")


# ── Synthetic Training Data ───────────────────────────────────────────────────

TRAINING_DATA: List[Tuple[str, str]] = [
    # Artificial Intelligence
    ("artificial intelligence reasoning knowledge representation expert systems planning", "Artificial Intelligence"),
    ("AI agents autonomous decision making intelligent systems logic inference", "Artificial Intelligence"),
    ("search algorithms heuristics problem solving AI planning knowledge base", "Artificial Intelligence"),
    ("intelligent agents environment perception action reward AI systems", "Artificial Intelligence"),
    ("automated reasoning symbolic AI ontology semantic knowledge graph", "Artificial Intelligence"),
    ("AI ethics fairness bias transparency explainability responsible AI", "Artificial Intelligence"),

    # Machine Learning
    ("machine learning supervised learning classification regression training data", "Machine Learning"),
    ("gradient descent optimization loss function model training neural network", "Machine Learning"),
    ("decision tree random forest ensemble methods boosting bagging", "Machine Learning"),
    ("support vector machine kernel methods feature space classification", "Machine Learning"),
    ("overfitting regularization cross validation hyperparameter tuning", "Machine Learning"),
    ("unsupervised learning clustering dimensionality reduction PCA k-means", "Machine Learning"),
    ("reinforcement learning reward policy agent environment Q-learning", "Machine Learning"),

    # Computer Vision
    ("image recognition convolutional neural network object detection visual", "Computer Vision"),
    ("image segmentation feature extraction CNN deep learning visual recognition", "Computer Vision"),
    ("object detection YOLO bounding box image classification visual features", "Computer Vision"),
    ("face recognition biometric image processing pixel feature map pooling", "Computer Vision"),
    ("medical imaging X-ray MRI scan segmentation diagnostic computer vision", "Computer Vision"),
    ("optical flow tracking video analysis frame temporal visual motion", "Computer Vision"),

    # Natural Language Processing
    ("natural language processing text classification sentiment analysis NLP", "Natural Language Processing"),
    ("transformer attention mechanism BERT GPT language model tokenization", "Natural Language Processing"),
    ("named entity recognition part of speech tagging text parsing NLP", "Natural Language Processing"),
    ("machine translation word embeddings word2vec semantic similarity", "Natural Language Processing"),
    ("question answering reading comprehension text understanding language", "Natural Language Processing"),
    ("text generation language model fine-tuning prompt tokens vocabulary", "Natural Language Processing"),
    ("speech recognition acoustic model phoneme transcription language model", "Natural Language Processing"),

    # Robotics
    ("robotics motion planning kinematics sensors actuators autonomous", "Robotics"),
    ("robot manipulation grasping inverse kinematics control feedback", "Robotics"),
    ("autonomous navigation SLAM localization mapping path planning robot", "Robotics"),
    ("humanoid robot bipedal locomotion balance control dynamics", "Robotics"),
    ("drone UAV aerial navigation obstacle avoidance control system", "Robotics"),
    ("human robot interaction collaborative robot safety perception", "Robotics"),

    # Cyber Security
    ("cybersecurity intrusion detection malware encryption vulnerability", "Cyber Security"),
    ("network security firewall threat detection anomaly authentication", "Cyber Security"),
    ("cryptography public key encryption digital signature hash function", "Cyber Security"),
    ("penetration testing vulnerability assessment exploit security audit", "Cyber Security"),
    ("ransomware phishing social engineering attack defense security", "Cyber Security"),
    ("zero trust security identity access management authentication authorization", "Cyber Security"),

    # Cloud Computing
    ("cloud computing distributed systems scalability microservices AWS", "Cloud Computing"),
    ("kubernetes docker container orchestration deployment cloud native", "Cloud Computing"),
    ("serverless functions lambda cloud architecture scalability", "Cloud Computing"),
    ("cloud storage database replication consistency availability partition", "Cloud Computing"),
    ("DevOps CI CD pipeline infrastructure as code terraform automation", "Cloud Computing"),
    ("edge computing fog computing latency distributed cloud IoT", "Cloud Computing"),
]


class DocumentClassifier:

    def __init__(self):
        self.model: Optional[object]   = None
        self.tokenizer: Optional[dict] = None
        self.categories = settings.CATEGORIES
        self.word_index: Dict[str, int] = {}
        self._load()

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def build_tokenizer(self, texts: List[str]) -> Dict[str, int]:
        """Build a simple word-index vocabulary."""
        word_counts: Dict[str, int] = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        word_index = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in sorted_words[:settings.MAX_VOCAB_SIZE - 2]:
            word_index[word] = len(word_index)
        return word_index

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        sequences = []
        for text in texts:
            seq = [self.word_index.get(word, 1) for word in text.split()]
            sequences.append(seq)
        return sequences

    def pad(self, sequences: List[List[int]]) -> np.ndarray:
        if not TF_AVAILABLE:
            # Manual padding
            result = np.zeros((len(sequences), settings.MAX_SEQUENCE_LENGTH), dtype=np.int32)
            for i, seq in enumerate(sequences):
                trunc = seq[:settings.MAX_SEQUENCE_LENGTH]
                result[i, :len(trunc)] = trunc
            return result
        return pad_sequences(
            sequences,
            maxlen=settings.MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post"
        )

    # ── Model Architecture ────────────────────────────────────────────────────

    def build_model(self, vocab_size: int, num_classes: int) -> "tf.keras.Model":
        model = Sequential([
            Embedding(vocab_size, 64, input_length=settings.MAX_SEQUENCE_LENGTH),
            GlobalAveragePooling1D(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation="softmax"),
        ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not installed.")

        logger.info("Starting TensorFlow classifier training...")

        texts  = [self.preprocess_text(t) for t, _ in TRAINING_DATA]
        labels = [self.categories.index(l) for _, l in TRAINING_DATA]

        self.word_index = self.build_tokenizer(texts)
        sequences = self.texts_to_sequences(texts)
        X = self.pad(sequences)
        y = np.array(labels)

        # Train/validation split
        split = int(0.85 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        vocab_size  = min(len(self.word_index), settings.MAX_VOCAB_SIZE)
        num_classes = len(self.categories)

        self.model = self.build_model(vocab_size, num_classes)

        history = self.model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=8,
            validation_data=(X_val, y_val),
            verbose=0
        )

        final_acc = history.history["accuracy"][-1]
        val_acc   = history.history.get("val_accuracy", [0])[-1]
        logger.info(f"Training complete — Accuracy: {final_acc:.3f} | Val Accuracy: {val_acc:.3f}")

        self._save()
        return {"train_accuracy": float(final_acc), "val_accuracy": float(val_acc)}

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(settings.MODEL_DIR, exist_ok=True)
        self.model.save(settings.ML_MODEL_FILE)
        with open(settings.TOKENIZER_FILE, "w") as f:
            json.dump(self.word_index, f)
        logger.info(f"Model saved to {settings.ML_MODEL_FILE}")

    def _load(self):
        if not TF_AVAILABLE:
            return
        if os.path.exists(settings.ML_MODEL_FILE) and os.path.exists(settings.TOKENIZER_FILE):
            try:
                self.model = load_model(settings.ML_MODEL_FILE)
                with open(settings.TOKENIZER_FILE, "r") as f:
                    self.word_index = json.load(f)
                logger.info("Classifier model loaded from disk.")
            except Exception as e:
                logger.warning(f"Could not load classifier model: {e}")

    def is_trained(self) -> bool:
        return self.model is not None and bool(self.word_index)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, text: str) -> Dict:
        if not self.is_trained():
            raise RuntimeError("Model not trained yet. Call /ml/train first.")

        cleaned = self.preprocess_text(text[:3000])
        seq     = self.texts_to_sequences([cleaned])
        X       = self.pad(seq)

        probs = self.model.predict(X, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        category   = self.categories[pred_idx]

        all_scores = {
            cat: round(float(probs[i]), 4)
            for i, cat in enumerate(self.categories)
        }

        logger.info(f"Classification: {category} ({confidence:.3f})")
        return {
            "category": category,
            "confidence": round(confidence, 4),
            "all_scores": all_scores
        }


# Singleton
document_classifier = DocumentClassifier()
