# ============================================================
# ENTERPRISE FEEDBACK ANALYTICS PIPELINE
# ============================================================
#
# FINAL ARCHITECTURE
# -------------------
#
# INTENT DETECTION:
#   -> Best pretrained embedding model
#   -> Semantic similarity
#   -> L1 + L2 retrieval
#
# SENTIMENT CLASSIFICATION:
#   -> YOUR finetuned LoRA model
#
# ============================================================



# ============================================================
# PROJECT STRUCTURE
# ============================================================

"""
feedback_ai/

│
├── data/
│   ├── intents.csv
│   ├── channel.csv
│   ├── digital.csv
│   ├── after_chat.csv
│   └── brand.csv
│
├── models/
│   ├── sentiment_base_model/
│   └── sentiment_lora_adapter/
│
├── outputs/
│
├── src/
│   ├── config.py
│   ├── embedding_model.py
│   ├── sentiment_model.py
│   ├── intent_loader.py
│   ├── vector_store.py
│   ├── predictor.py
│   ├── dataset_runner.py
│   └── main.py
│
└── requirements.txt
"""



# ============================================================
# requirements.txt
# ============================================================

"""
sentence-transformers
transformers
peft
torch
pandas
numpy
scikit-learn
tqdm
"""



# ============================================================
# SAMPLE intents.csv
# ============================================================

"""
l1,l2,description

Network Issue,Slow Speed,Customer facing slow internet speed

Network Issue,Frequent Disconnection,Internet disconnecting frequently

Billing,Incorrect Charges,Wrong billing amount charged

Customer Service,Rude Agent,Support agent behaving rudely

App Experience,Login Issue,Unable to login into application
"""



# ============================================================
# src/config.py
# ============================================================

# ------------------------------------------------------------
# BEST EMBEDDING MODEL FOR INTENT SEARCH
# ------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# ------------------------------------------------------------
# YOUR SENTIMENT MODEL
# ------------------------------------------------------------

SENTIMENT_BASE_MODEL = "../models/sentiment_base_model"

SENTIMENT_LORA_PATH = "../models/sentiment_lora_adapter"

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

INTENT_CSV_PATH = "../data/intents.csv"

SIMILARITY_THRESHOLD = 0.65

DEVICE = "cuda"



# ============================================================
# src/embedding_model.py
# ============================================================

from sentence_transformers import SentenceTransformer


class IntentEmbeddingModel:

    def __init__(self, model_name):

        self.model = SentenceTransformer(model_name)

    def encode(self, texts):

        return self.model.encode(

            texts,

            normalize_embeddings=True,

            convert_to_numpy=True
        )



# ============================================================
# src/sentiment_model.py
# ============================================================

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from peft import PeftModel

from scipy.special import softmax

from config import (
    SENTIMENT_BASE_MODEL,
    SENTIMENT_LORA_PATH
)


class SentimentClassifier:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            SENTIMENT_BASE_MODEL
        )

        base_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                SENTIMENT_BASE_MODEL
            )
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            SENTIMENT_LORA_PATH
        )

        self.model.eval()

        self.id2label = (
            self.model.config.id2label
        )

    # --------------------------------------------------------
    # SENTIMENT PREDICTION
    # --------------------------------------------------------

    def predict(self, text):

        inputs = self.tokenizer(

            text,

            truncation=True,

            padding=True,

            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = self.model(**inputs)

        logits = outputs.logits[0]

        probs = softmax(
            logits.cpu().numpy()
        )

        predicted_idx = probs.argmax()

        return {

            "sentiment":
                self.id2label[predicted_idx],

            "sentiment_score":
                float(probs[predicted_idx])
        }



# ============================================================
# src/intent_loader.py
# ============================================================

import pandas as pd


class IntentLoader:

    def __init__(self, csv_path):

        self.csv_path = csv_path

    def load(self):

        df = pd.read_csv(self.csv_path)

        return df



# ============================================================
# src/vector_store.py
# ============================================================

class VectorStore:

    def __init__(self):

        self.records = []

    # --------------------------------------------------------
    # BUILD VECTOR DATABASE
    # --------------------------------------------------------

    def build(

        self,

        intent_df,

        embedding_model
    ):

        texts = []

        for _, row in intent_df.iterrows():

            text = (

                f"L1 Intent: {row['l1']} "

                f"L2 Intent: {row['l2']} "

                f"Description: {row['description']}"
            )

            texts.append(text)

        embeddings = embedding_model.encode(texts)

        for idx, row in intent_df.iterrows():

            self.records.append({

                "l1": row["l1"],

                "l2": row["l2"],

                "description":
                    row["description"],

                "embedding":
                    embeddings[idx]
            })



# ============================================================
# src/predictor.py
# ============================================================

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from config import (
    EMBEDDING_MODEL_NAME,
    INTENT_CSV_PATH,
    SIMILARITY_THRESHOLD
)

from embedding_model import IntentEmbeddingModel

from sentiment_model import SentimentClassifier

from intent_loader import IntentLoader

from vector_store import VectorStore


class FeedbackPredictor:

    def __init__(self):

        # ----------------------------------------------------
        # INTENT EMBEDDING MODEL
        # ----------------------------------------------------

        self.embedding_model = (
            IntentEmbeddingModel(
                EMBEDDING_MODEL_NAME
            )
        )

        # ----------------------------------------------------
        # LOAD INTENT CSV
        # ----------------------------------------------------

        loader = IntentLoader(
            INTENT_CSV_PATH
        )

        intent_df = loader.load()

        # ----------------------------------------------------
        # VECTOR STORE
        # ----------------------------------------------------

        self.vector_store = VectorStore()

        self.vector_store.build(

            intent_df,

            self.embedding_model
        )

        # ----------------------------------------------------
        # SENTIMENT MODEL
        # ----------------------------------------------------

        self.sentiment_model = (
            SentimentClassifier()
        )

    # --------------------------------------------------------
    # INTENT PREDICTION
    # --------------------------------------------------------

    def predict_intent(self, text):

        query_embedding = (
            self.embedding_model
            .encode([text])[0]
        )

        embeddings = [

            r["embedding"]

            for r in self.vector_store.records
        ]

        similarities = cosine_similarity(

            [query_embedding],

            embeddings
        )[0]

        best_idx = np.argmax(similarities)

        best_record = (
            self.vector_store.records[best_idx]
        )

        confidence = (
            float(similarities[best_idx])
        )

        if confidence < SIMILARITY_THRESHOLD:

            return {

                "predicted_l1":
                    "Unknown",

                "predicted_l2":
                    "Unknown",

                "intent_confidence":
                    confidence
            }

        return {

            "predicted_l1":
                best_record["l1"],

            "predicted_l2":
                best_record["l2"],

            "intent_confidence":
                confidence
        }

    # --------------------------------------------------------
    # COMPLETE PREDICTION
    # --------------------------------------------------------

    def predict(self, text):

        intent_result = (
            self.predict_intent(text)
        )

        sentiment_result = (
            self.sentiment_model
            .predict(text)
        )

        return {

            "feedback": text,

            "predicted_l1":
                intent_result["predicted_l1"],

            "predicted_l2":
                intent_result["predicted_l2"],

            "intent_confidence":
                intent_result["intent_confidence"],

            "sentiment":
                sentiment_result["sentiment"],

            "sentiment_score":
                sentiment_result["sentiment_score"]
        }



# ============================================================
# src/dataset_runner.py
# ============================================================

import pandas as pd

from tqdm import tqdm

from predictor import FeedbackPredictor


class DatasetRunner:

    def __init__(self):

        self.predictor = (
            FeedbackPredictor()
        )

    # --------------------------------------------------------
    # PROCESS DATASET
    # --------------------------------------------------------

    def run_dataset(

        self,

        input_csv,

        output_csv,

        text_columns
    ):

        df = pd.read_csv(input_csv)

        results = []

        for _, row in tqdm(
            df.iterrows(),
            total=len(df)
        ):

            combined_text = ""

            # ---------------------------------------------
            # MERGE MULTIPLE TEXT COLUMNS
            # ---------------------------------------------

            for col in text_columns:

                if (
                    col in row
                    and pd.notna(row[col])
                ):

                    combined_text += (
                        str(row[col]) + " "
                    )

            combined_text = (
                combined_text.strip()
            )

            prediction = (
                self.predictor
                .predict(combined_text)
            )

            result_row = row.to_dict()

            result_row.update(prediction)

            results.append(result_row)

        result_df = pd.DataFrame(results)

        result_df.to_csv(
            output_csv,
            index=False
        )

        print(f"Saved: {output_csv}")



# ============================================================
# src/main.py
# ============================================================

from dataset_runner import DatasetRunner

runner = DatasetRunner()


# ============================================================
# CHANNEL SURVEY
# ============================================================

runner.run_dataset(

    input_csv="../data/channel.csv",

    output_csv="../outputs/channel_output.csv",

    text_columns=[
        "feedback",
        "additional_comments"
    ]
)


# ============================================================
# DIGITAL SURVEY
# ============================================================

runner.run_dataset(

    input_csv="../data/digital.csv",

    output_csv="../outputs/digital_output.csv",

    text_columns=[
        "survey_text"
    ]
)


# ============================================================
# AFTER CHAT
# ============================================================

runner.run_dataset(

    input_csv="../data/after_chat.csv",

    output_csv="../outputs/after_chat_output.csv",

    text_columns=[
        "chat_feedback",
        "agent_comments"
    ]
)


# ============================================================
# BRAND FEEDBACK
# ============================================================

runner.run_dataset(

    input_csv="../data/brand.csv",

    output_csv="../outputs/brand_output.csv",

    text_columns=[
        "brand_feedback"
    ]
)



# ============================================================
# OUTPUT COLUMNS
# ============================================================

"""
feedback
predicted_l1
predicted_l2
intent_confidence
sentiment
sentiment_score
"""



# ============================================================
# HOW TO RUN
# ============================================================

# STEP 1
# Install packages

"""
pip install -r requirements.txt
"""


# STEP 2
# Run pipeline

"""
python main.py
"""



# ============================================================
# WHY THIS ARCHITECTURE IS STRONG
# ============================================================

"""
INTENT DETECTION
----------------
Uses best semantic embedding model:
- BGE Large
- Better semantic retrieval
- Better intent matching
- No overfitting issues


SENTIMENT
----------
Uses YOUR custom LoRA model:
- Domain-specific learning
- Telecom sentiment understanding
- Your finetuned behavior


BEST OF BOTH WORLDS
-------------------
Intent:
    Retrieval-based semantic system

Sentiment:
    Finetuned classifier
"""
