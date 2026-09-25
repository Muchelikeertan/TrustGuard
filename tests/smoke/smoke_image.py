import os
import sys

# Windows UTF-8 console output encoding fix for Unicode/Emoji in DeepFace logger
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Required legacy Keras configuration for TensorFlow 2.16+ / DeepFace compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# Suppress excessive TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path


def run_smoke_image(
    img_a: str = "sample_data/images/same_person_a.jpg",
    img_b: str = "sample_data/images/same_person_b.jpg",
    img_diff: str = "sample_data/images/different_person.jpg",
) -> dict:
    """Verifies face consistency using DeepFace.
    
    Returns:
        dict: {'score': float or None, 'label': str, 'reason': str}
    """
    path_a = Path(img_a)
    path_b = Path(img_b)
    path_diff = Path(img_diff)

    missing = []
    for p in [path_a, path_b, path_diff]:
        if not p.exists():
            missing.append(str(p))

    if missing:
        print("[WAITING FOR DATA] Missing required sample image file(s):")
        for m in missing:
            print(f"  -> Please add: {m}")
        return {
            "score": None,
            "label": "WAITING FOR DATA",
            "reason": f"Missing image files: {', '.join(missing)}",
        }

    # Import DeepFace after environment variables are set
    from deepface import DeepFace

    print(f"--- Running DeepFace Verification ---")
    print(f"Comparing pair 1 (Same Person): {path_a} vs {path_b}")
    res_same = DeepFace.verify(
        img1_path=str(path_a),
        img2_path=str(path_b),
        enforce_detection=False,
    )
    print(f"  Verified: {res_same.get('verified')}")
    print(f"  Distance: {res_same.get('distance'):.4f}")
    print(f"  Threshold: {res_same.get('threshold'):.4f}")

    print(f"Comparing pair 2 (Different Person): {path_a} vs {path_diff}")
    res_diff = DeepFace.verify(
        img1_path=str(path_a),
        img2_path=str(path_diff),
        enforce_detection=False,
    )
    print(f"  Verified: {res_diff.get('verified')}")
    print(f"  Distance: {res_diff.get('distance'):.4f}")
    print(f"  Threshold: {res_diff.get('threshold'):.4f}")

    # Compute confidence/consistency score based on same pair matching and diff pair divergence
    same_verified = res_same.get("verified", False)
    diff_verified = res_diff.get("verified", False)
    dist_same = res_same.get("distance", 1.0)
    thresh = res_same.get("threshold", 0.6)

    # Score: 1.0 means high consistency (same person confirmed, different person rejected)
    if same_verified and not diff_verified:
        score = max(0.0, min(1.0, 1.0 - (dist_same / thresh) * 0.5))
        label = "CONSISTENT"
        reason = f"Same-person matched (dist={dist_same:.3f}) and different-person rejected (dist={res_diff.get('distance'):.3f})."
    else:
        score = 0.2
        label = "INCONSISTENT"
        reason = f"Mismatch observed: same_verified={same_verified}, diff_verified={diff_verified}."

    return {
        "score": score,
        "label": label,
        "reason": reason,
    }


if __name__ == "__main__":
    result = run_smoke_image()
    print("\nResult:", result)
