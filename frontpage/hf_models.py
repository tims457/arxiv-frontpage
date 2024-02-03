import os
from numpy import dot
from openai import OpenAI
from typing import List, Dict
from rich.console import Console
from transformers import pipeline

from dotenv import load_dotenv
from .constants import (
    DATA_LEVELS,
    INDICES_FOLDER,
    LABELS,
    EMBED_LABELS,
    CONFIG,
    THRESHOLDS,
    MODEL_TYPE,
)

load_dotenv()
console = Console()


class Model:
    def __init__(self) -> None:
        if MODEL_TYPE == "classifier":
            console.log("Loading classifier model.")
            self.classifier = pipeline(
                "zero-shot-classification", model="facebook/bart-large-mnli"
            )
            console.log("Model loaded.")
        elif MODEL_TYPE == "embedding":
            console.log("Using OPENAI API for embeddings.")
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            self.label_embeddings = {
                label: self.openai_client.embeddings.create(
                    input=embed_label, model="text-embedding-ada-002"
                )
                .data[0]
                .embedding
                for label, embed_label in zip(LABELS, EMBED_LABELS)
            }

    def embed(self, text: str):
        return (
            self.openai_client.embeddings.create(
                input=text, model="text-embedding-ada-002"
            )
            .data[0]
            .embedding
        )

    def predict(self, sentences: List[str]):
        console.log(f"Predicting labels for:\n\t\t{(' ').join(sentences[:3])}")
        result = [{} for _ in sentences]
        all_probs = self.classifier(sentences, LABELS, multi_label=True)

        for i, probs in enumerate(all_probs):
            for j, label in enumerate(probs["labels"]):
                result[i][label] = probs["scores"][j]
        return result

    def query(self, payload):
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}

        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    def predict_api(self, sentences: List[str]):
        console.log(f"Predicting labels for:\n\t\t{(' ').join(sentences[:3])}")
        result = [{} for _ in sentences]
        payload = {
            "inputs": sentences,
            "parameters": {
                "candidate_labels": LABELS,
                "multi_label": True,
            },
        }
        all_probs = self.query(payload)
        for i, probs in enumerate(all_probs):
            for j, label in enumerate(probs["labels"]):
                result[i][label] = probs["scores"][j]
        return result

    def predict_embeddings(self, sentences: List[str]):
        console.log(f"Predicting labels for:\n\t\t{(' ').join(sentences[:3])}")
        result = [{} for _ in sentences]

        # Get the embeddings
        response = self.openai_client.embeddings.create(
            input=sentences, model="text-embedding-ada-002"
        )

        # The embeddings are in the response
        embeddings = [d.embedding for d in response.data]
        for i, embedding in enumerate(embeddings):
            for label, label_embedding in self.label_embeddings.items():
                result[i][label] = dot(embedding, label_embedding)

        return result
