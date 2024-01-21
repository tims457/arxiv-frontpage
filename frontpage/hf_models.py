from typing import List, Dict
from rich.console import Console 
from transformers import pipeline

from .constants import DATA_LEVELS, INDICES_FOLDER, LABELS, CONFIG, THRESHOLDS


console = Console()



class HFModel:
    def __init__(self) -> None:
        
        console.log("Loading model.")
        #1
        # self.classifier = pipeline("zero-shot-classification",
        #             model="knowledgator/comprehend_it-base")
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        #3
        # self.classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33")
        
        console.log("Model loaded.")
        
    def predict(self, sentences: List[str]):
        # candidate_labels = ['travel', 'cooking', 'dancing', 'exploration']
        console.log(f"Predicting labels for:\n\t\t{(' ').join(sentences[:3])}")
        result = [{} for _ in sentences]
        all_probs = self.classifier(sentences, LABELS, multi_label=True)
        # for i, sentence in enumerate(sentences):
        #     probs = self.classifier(sentence, LABELS, multi_label=True)
        for i, probs in enumerate(all_probs):
            for j, label in enumerate(probs['labels']):
                result[i][label] = probs['scores'][j]
        return result        

    def predict2(self, sentences: List[str]):
        console.log(f"Predicting labels for:\n\t\t{(' ').join(sentences[:3])}")
        result = [{} for _ in sentences]
        
        probs = self.classifier((' ').join(sentences), LABELS, multi_label=True)
        for label, prob in zip(LABELS, probs['scores']):
            result[label] = prob
        return result      
    
    
    def predict3(self, sentences: List[str]):
        hypothesis_template = "This example is about {}"
        # output = zeroshot_classifier(text, LABELS, hypothesis_template=hypothesis_template, multi_label=True)
        console.log(f"Predicting labels for:\n\t\t{(' ').join(sentences[:3])}")
        result = [{} for _ in sentences]
        for i, sentence in enumerate(sentences):
            probs = self.classifier(sentence, LABELS, hypothesis_template=hypothesis_template, multi_label=True)
            for label, prob in zip(LABELS, probs['scores']):
                result[i][label] = prob
        return result 

        

