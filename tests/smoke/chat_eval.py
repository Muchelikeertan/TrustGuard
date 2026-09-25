"""Evaluation script for TrustGuard chat analysis.

Evaluates 20 benign and 20 scam messages across English and Hinglish,
including edge cases with benign urgency/transfer keywords and subtle scam tactics.
Loads the model once and determines optimal classification thresholds.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from trustguard.chat_check import analyze_chat, get_classifier

# Dataset of 20 Scam and 20 Benign messages
DATASET = [
    # --- 20 SCAM MESSAGES ---
    # 1. User specified anchor scam
    {"text": "I'm Rahul, don't call me, send Rs 20,000 now", "is_scam": True, "category": "impersonation_urgency"},
    # 2-5: Hinglish Scams (4 of 8 total Hinglish)
    {"text": "Bhai mera accident ho gaya hai, turant 10000 bhejo is UPI pe, call mat karna hospital me hu.", "is_scam": True, "category": "hinglish_emergency"},
    {"text": "Aapka bank account suspend ho jayega, abhi ke abhi apna OTP aur PIN share karo to reactivate.", "is_scam": True, "category": "hinglish_credential_theft"},
    {"text": "Yeh secret deal hai kisi ko mat batana, fatafat 50000 transfer kar do crypto wallet me double profit hoga.", "is_scam": True, "category": "hinglish_investment_scam"},
    {"text": "Customs department se bol raha hu, aapke parcel me drugs mila hai, police aayegi agar 25000 fine nahi bhara.", "is_scam": True, "category": "hinglish_digital_arrest"},
    # 6-10: Subtle scams avoiding obvious keywords (5 subtle scams)
    {"text": "Hey mom, my phone fell in water and won't turn on, borrowing someone's device to ask you to settle my hotel room fee so I don't get kicked out: http://bit.ly/hotel-pay", "is_scam": True, "category": "subtle_family_impersonation"},
    {"text": "Hello, regarding your shortlisted job application, your onboarding background screening verification fee is pending before the offer letter can be released.", "is_scam": True, "category": "subtle_job_scam"},
    {"text": "Hi, your nephew gave me your number. He is stuck at the interstate toll booth without his wallet, could you help him clear the gate fee via gpay?", "is_scam": True, "category": "subtle_relative_impersonation"},
    {"text": "Your residential electricity service will be disconnected at midnight tonight due to an unpaid discrepancy. Settle immediately at bit.ly/power-bill-fix", "is_scam": True, "category": "subtle_utility_phishing"},
    {"text": "Hey team lead, I am in an executive meeting and cannot speak on the phone. Can you purchase three Amazon gift cards for the client presentation?", "is_scam": True, "category": "subtle_ceo_fraud"},
    # 11-20: High pressure, threats, OTP requests, and financial coercion
    {"text": "Urgent! Your debit card is blocked. Send the 6 digit verification passcode received on your mobile to restore access immediately.", "is_scam": True, "category": "otp_phishing"},
    {"text": "CBI Cyber Crime Cell: A non-bailable warrant has been issued in your name. Do not contact family members, pay the bond penalty right now.", "is_scam": True, "category": "law_enforcement_impersonation"},
    {"text": "Emergency! I lost my wallet in London and need wire transfer of $500 for medication. Please do not tell dad, he will panic.", "is_scam": True, "category": "distress_scam"},
    {"text": "Income tax evasion detected on your PAN. Immediate legal action will be taken unless Rs 45,000 is deposited to avoid custody.", "is_scam": True, "category": "tax_threat"},
    {"text": "Congratulations! You won Rs 50,00,000 in lottery. To release the winnings, share your net-banking password and CVV.", "is_scam": True, "category": "lottery_scam"},
    {"text": "Dear customer, your KYC is expired. Update within 15 minutes by clicking wa.me/919988776655 or banking service terminates.", "is_scam": True, "category": "kyc_phishing"},
    {"text": "I know what you did on your webcam last night. Pay 0.5 Bitcoin to my wallet right now or the video goes to all your contacts.", "is_scam": True, "category": "sextortion_blackmail"},
    {"text": "Courier alert: FedEx parcel held at Mumbai airport. Pay custom clearance Rs 3,400 right now or face police investigation.", "is_scam": True, "category": "parcel_scam"},
    {"text": "Aapke credit card reward points expire ho rahe hain. Redeem karne ke liye apna card number aur CVV daalo.", "is_scam": True, "category": "reward_points_scam"},
    {"text": "Police complaint darj ho chuki hai aapke naam par. Court warrant se bachne ke liye turant legal fee jama karein.", "is_scam": True, "category": "warrant_threat"},

    # --- 20 BENIGN MESSAGES ---
    # 21-24: Hinglish Benign (4 of 8 total Hinglish)
    {"text": "Bhai kal shaam ko match dekhne chalein kya? Mujhe 6 baje call kar lena.", "is_scam": False, "category": "hinglish_casual"},
    {"text": "Mummy maine khana kha liya hai, abhi library ja raha hu padhai karne.", "is_scam": False, "category": "hinglish_family"},
    {"text": "Project ka documentation complete ho gaya hai, kal sir ko submit kar denge.", "is_scam": False, "category": "hinglish_work"},
    {"text": "Diwali ki chuttiyo me sab ghar aa rahe hain, train ki ticket book kar li maine.", "is_scam": False, "category": "hinglish_holiday"},
    # 25-29: Benign messages containing "urgent" or "transfer" in harmless contexts (5 harmless keywords)
    {"text": "I am submitting the urgent assignment before the midnight deadline, hope professor accepts it.", "is_scam": False, "category": "benign_urgent_academic"},
    {"text": "The doctor mentioned that grandfather needs urgent bed rest for the next three days.", "is_scam": False, "category": "benign_urgent_medical"},
    {"text": "Great news! I finally got a transfer to our Bangalore tech branch starting next month.", "is_scam": False, "category": "benign_transfer_job"},
    {"text": "Could you transfer the meeting minutes doc from your Google Drive to our shared folder?", "is_scam": False, "category": "benign_transfer_file"},
    {"text": "Urgent update: our commuter train has been delayed by 20 minutes due to fog, see you soon.", "is_scam": False, "category": "benign_urgent_transit"},
    # 30-40: Everyday casual, work, personal, and polite conversation
    {"text": "Hey, are you free for lunch tomorrow at the cafeteria around 1 PM?", "is_scam": False, "category": "casual_social"},
    {"text": "Hi Mom, just landed safely at the airport. Taking an Uber home now.", "is_scam": False, "category": "family_update"},
    {"text": "Thanks for sending over the project report, I will review the slides this evening.", "is_scam": False, "category": "work_review"},
    {"text": "Don't forget to bring the science textbook to class tomorrow morning.", "is_scam": False, "category": "school_reminder"},
    {"text": "Happy birthday Sarah! Wishing you a wonderful year ahead filled with happiness and joy.", "is_scam": False, "category": "greeting"},
    {"text": "Let's reschedule our sync to Thursday at 3 PM if that works better for the design team.", "is_scam": False, "category": "work_schedule"},
    {"text": "Can someone please share the guest WiFi password for the 4th floor conference room?", "is_scam": False, "category": "it_support"},
    {"text": "The recipe calls for 2 cups of flour, fresh cream, and one teaspoon of vanilla extract.", "is_scam": False, "category": "hobby_cooking"},
    {"text": "We should plan a weekend hike at the national park once the weather clears up.", "is_scam": False, "category": "weekend_activity"},
    {"text": "Please review the attached invoice for last month's cloud hosting services when you have time.", "is_scam": False, "category": "accounting_invoice"},
    {"text": "Good morning team, our sprint planning session will begin at 10 AM in Room 302.", "is_scam": False, "category": "work_morning"},
]


def evaluate_dataset():
    """Runs batch evaluation on the 40 test messages."""
    print("=" * 80)
    print("TRUSTGUARD CHAT ANALYSIS EVALUATION SUITE")
    print(f"Total messages: {len(DATASET)} (20 Scam, 20 Benign)")
    print("Pre-loading model into memory once...")
    print("=" * 80)

    # Ensure model is warmed up once
    get_classifier()

    results = []

    for idx, item in enumerate(DATASET, 1):
        msg = item["text"]
        is_scam = item["is_scam"]
        category = item["category"]

        signal = analyze_chat(msg)

        score = signal.score if signal.score is not None else 0.0
        status = signal.status
        model_score = signal.details.get("model_score", 0.0)
        rule_score = signal.details.get("rule_score", 0.0)
        fired_rules = [r["rule"] for r in signal.details.get("fired_rules", [])]

        results.append({
            "idx": idx,
            "text": msg,
            "is_scam": is_scam,
            "category": category,
            "score": score,
            "status": status,
            "model_score": model_score,
            "rule_score": rule_score,
            "fired_rules": fired_rules,
        })

        expected_type = "SCAM" if is_scam else "BENIGN"
        flag = "[OK]" if (is_scam and status in ("SUSPICIOUS", "HIGH_PRESSURE")) or (not is_scam and status == "NORMAL") else "[MISMATCH]"
        
        print(f"\n#{idx:02d} [{expected_type}] {flag} Score: {score:.4f} | Status: {status}")
        print(f"  Message: \"{msg}\"")
        print(f"  Model Score: {model_score:.4f} | Rule Score: {rule_score:.4f} | Fired Rules: {fired_rules}")

    # --- Metrics Computation under Default Thresholds ---
    scams_flagged = sum(1 for r in results if r["is_scam"] and r["status"] in ("SUSPICIOUS", "HIGH_PRESSURE"))
    benign_flagged = sum(1 for r in results if not r["is_scam"] and r["status"] in ("SUSPICIOUS", "HIGH_PRESSURE"))
    scams_total = sum(1 for r in results if r["is_scam"])
    benign_total = sum(1 for r in results if not r["is_scam"])

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY (Default Thresholds)")
    print("=" * 80)
    print(f"Scams Flagged (True Positives):       {scams_flagged} / {scams_total} ({scams_flagged/scams_total*100:.1f}%)")
    print(f"Benign Flagged (False Positives):     {benign_flagged} / {benign_total} ({benign_flagged/benign_total*100:.1f}%)")
    print(f"Clean Benign (True Negatives):        {benign_total - benign_flagged} / {benign_total}")
    print(f"Missed Scams (False Negatives):       {scams_total - scams_flagged} / {scams_total}")

    # --- Threshold Grid Search to Find Optimal Separation ---
    print("\n" + "-" * 80)
    print("THRESHOLD OPTIMIZATION GRID SEARCH")
    print("-" * 80)

    best_f1 = -1.0
    best_suspicious = 0.55
    best_high_pressure = 0.80
    best_metrics = {}

    # Grid search across potential threshold pairings
    for s_step in range(35, 75, 5):
        s_thresh = round(s_step / 100.0, 2)
        for h_step in range(max(s_step + 10, 70), 95, 5):
            h_thresh = round(h_step / 100.0, 2)

            tp = sum(1 for r in results if r["is_scam"] and r["score"] >= s_thresh)
            fp = sum(1 for r in results if not r["is_scam"] and r["score"] >= s_thresh)
            fn = sum(1 for r in results if r["is_scam"] and r["score"] < s_thresh)
            tn = sum(1 for r in results if not r["is_scam"] and r["score"] < s_thresh)

            # High pressure subset of scams
            hp_scams = sum(1 for r in results if r["is_scam"] and r["score"] >= h_thresh)
            hp_benign = sum(1 for r in results if not r["is_scam"] and r["score"] >= h_thresh)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # We prioritize zero false positives on benign (fp == 0), then highest F1
            optimization_score = f1 - (fp * 0.1)

            if optimization_score > best_f1:
                best_f1 = optimization_score
                best_suspicious = s_thresh
                best_high_pressure = h_thresh
                best_metrics = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "hp_scams": hp_scams,
                    "hp_benign": hp_benign,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

    print(f"Optimal Thresholds Found:")
    print(f"  best suspicious_min:    {best_suspicious:.2f}")
    print(f"  best high_pressure_min: {best_high_pressure:.2f}")
    print(f"  Resulting Precision:    {best_metrics['precision']*100:.1f}%")
    print(f"  Resulting Recall:       {best_metrics['recall']*100:.1f}%")
    print(f"  Resulting F1 Score:     {best_metrics['f1']:.4f}")
    print(f"  False Positives:        {best_metrics['fp']}")
    print(f"  False Negatives:        {best_metrics['fn']}")
    print(f"  High Pressure Scams:    {best_metrics['hp_scams']} (Benign in High Pressure: {best_metrics['hp_benign']})")
    print("=" * 80)

    return {
        "best_suspicious_min": best_suspicious,
        "best_high_pressure_min": best_high_pressure,
        "scams_flagged": scams_flagged,
        "benign_flagged": benign_flagged,
    }


if __name__ == "__main__":
    evaluate_dataset()
