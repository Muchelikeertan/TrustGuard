import os
import sys

# Suppress symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def run_smoke_chat() -> dict:
    """Classifies sample chat messages for scam indicators using zero-shot classification.

    Returns:
        dict: {'score': float, 'label': str, 'reason': str}
    """
    from transformers import pipeline

    print("--- Loading Zero-Shot Classification Pipeline (facebook/bart-large-mnli) ---")
    try:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        print(f"[ERROR] Failed to load pipeline: {e}")
        return {
            "score": 0.0,
            "label": "FAIL",
            "reason": f"Pipeline initialization failed: {e}",
        }

    candidate_labels = [
        "urgent money request",
        "discourages verification",
        "normal conversation",
    ]

    scam_messages = [
        "I'm Rahul, don't call me, send Rs 20,000 now",
        "Emergency! My phone is broken and cannot receive calls. Transfer Rs 15,000 urgently to this UPI ID immediately.",
        "Urgent CBI notice, do not verify with anyone or talk to relatives, transfer the clearance penalty right now.",
    ]

    legit_messages = [
        "Hey, are you free for lunch tomorrow at the cafeteria around 1 PM?",
        "Hi Mom, just landed safely at the airport. Taking a cab home now.",
        "Thanks for sending over the project report, I will review the slides this evening.",
    ]

    print("\n--- Testing Suspicious / Scam Messages ---")
    scam_scores = []
    for msg in scam_messages:
        res = classifier(msg, candidate_labels=candidate_labels)
        top_label = res["labels"][0]
        top_score = res["scores"][0]
        # Get score for scam-indicative categories
        scores_map = dict(zip(res["labels"], res["scores"]))
        urgent_score = scores_map.get("urgent money request", 0.0)
        discourage_score = scores_map.get("discourages verification", 0.0)
        normal_score = scores_map.get("normal conversation", 0.0)
        risk_score = urgent_score + discourage_score
        scam_scores.append(risk_score)
        print(f"Message: \"{msg}\"")
        print(f"  Top Label: {top_label} ({top_score:.4f})")
        print(f"  Breakdown: urgent={urgent_score:.4f}, discourages_verif={discourage_score:.4f}, normal={normal_score:.4f}\n")

    print("--- Testing Legitimate Messages ---")
    legit_scores = []
    for msg in legit_messages:
        res = classifier(msg, candidate_labels=candidate_labels)
        top_label = res["labels"][0]
        top_score = res["scores"][0]
        scores_map = dict(zip(res["labels"], res["scores"]))
        normal_score = scores_map.get("normal conversation", 0.0)
        risk_score = scores_map.get("urgent money request", 0.0) + scores_map.get("discourages verification", 0.0)
        legit_scores.append(risk_score)
        print(f"Message: \"{msg}\"")
        print(f"  Top Label: {top_label} ({top_score:.4f})")
        print(f"  Breakdown: urgent={scores_map.get('urgent money request', 0.0):.4f}, discourages_verif={scores_map.get('discourages verification', 0.0):.4f}, normal={normal_score:.4f}\n")

    avg_scam_risk = sum(scam_scores) / len(scam_scores)
    avg_legit_risk = sum(legit_scores) / len(legit_scores)

    discrimination_score = max(0.0, min(1.0, avg_scam_risk - avg_legit_risk))

    return {
        "score": round(discrimination_score, 4),
        "label": "PASS",
        "reason": f"Zero-shot classification pipeline executed successfully. Scam detection confidence avg={avg_scam_risk:.3f}, legit risk avg={avg_legit_risk:.3f}.",
    }


if __name__ == "__main__":
    result = run_smoke_chat()
    print("Result:", result)
