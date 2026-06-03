from transformers import pipeline
import json, os

MODEL_REPO = os.getenv("MODEL_REPO", "G25AIT2100/sms-spam-distilbert")

classifier = pipeline("text-classification", model=MODEL_REPO)

with open("id2label.json") as f:
    id2label = json.load(f)

def predict(text: str) -> dict:
    result = classifier(text)[0]
    return {
        "text":  text,
        "label": result["label"],
        "score": round(result["score"], 4)
    }

if __name__ == "__main__":
    samples = [
        "Congratulations! You've won a free iPhone. Click here to claim.",
        "Hey, are you coming to the meeting tomorrow?",
        "WINNER!! You have been selected to receive a prize reward!",
        "Can you pick up some groceries on your way home?",
    ]
    print("=== Inference Results ===")
    for s in samples:
        print(predict(s))
