from typing_extensions import Tuple
def determine_emotion(text: str) -> Tuple[str, str, str]:
    from transformers import pipeline
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    outputs = classifier(text)
    return outputs[0][0]['label'], outputs[0][1]['label'], outputs[0][2]['label']  # type: ignore

determine_emotion('I am super gay for my friend asshole it\'s hot')