"""
advisory.py
-----------
Agent 5 — Citizen Health Risk Advisory System.

Turns Agent 1 (attribution) + Agent 2 (forecast) output into plain-language,
multi-language advisories. The production design uses an LLM layer over the
forecast outputs for natural, varied phrasing in five regional languages.
This prototype uses templated text instead, so it runs with zero extra API
keys — but the templates are structured so swapping in an LLM call later
(e.g. the Anthropic API) is a one-function change, not a rewrite.
"""

from typing import Dict, List

SEVERITY_BANDS = [
    (0, 50, "Good", "Air quality is good. Enjoy normal outdoor activity."),
    (51, 100, "Satisfactory", "Air quality is acceptable. Sensitive groups should watch for symptoms."),
    (101, 200, "Moderate", "Sensitive groups (children, elderly, asthma/heart patients) should limit prolonged outdoor exertion."),
    (201, 300, "Poor", "Avoid outdoor exercise. Sensitive groups should stay indoors where possible."),
    (301, 400, "Very Poor", "Avoid outdoor activity. Schools should consider limiting outdoor sessions."),
    (401, 1000, "Severe", "Health emergency for sensitive groups. Stay indoors, use an air purifier or N95 mask if going out."),
]

# Minimal static translation templates — production version would route this
# through the LLM advisory layer for natural phrasing and full language coverage.
LANGUAGE_TEMPLATES = {
    "English": {
        "header": "Air Quality Advisory — {ward}",
        "now": "Current AQI: {aqi} ({category}).",
        "tomorrow": "Tomorrow morning AQI is expected to be around {tomorrow_aqi}.",
        "advice": "{advice}",
    },
    "Hindi": {
        "header": "\u0935\u093e\u092f\u0941 \u0917\u0941\u0923\u0935\u0924\u094d\u0924\u093e \u0938\u0932\u093e\u0939 \u2014 {ward}",
        "now": "\u0935\u0930\u094d\u0924\u092e\u093e\u0928 AQI: {aqi} ({category}).",
        "tomorrow": "\u0915\u0932 \u0938\u0941\u092c\u0939 AQI \u0932\u0917\u092d\u0917 {tomorrow_aqi} \u0930\u0939\u0928\u0947 \u0915\u0940 \u0938\u0902\u092d\u093e\u0935\u0928\u093e \u0939\u0948\u0964",
        "advice": "{advice_hi}",
    },
}

ADVICE_HINDI = {
    "Good": "\u092c\u093e\u0939\u0930 \u0915\u0940 \u0917\u0924\u093f\u0935\u093f\u0927\u093f\u092f\u093e\u0902 \u0938\u093e\u092e\u093e\u0928\u094d\u092f \u0930\u0942\u092a \u0938\u0947 \u0915\u0930 \u0938\u0915\u0924\u0947 \u0939\u0948\u0902\u0964",
    "Satisfactory": "\u0938\u0902\u0935\u0947\u0926\u0928\u0936\u0940\u0932 \u0935\u0930\u094d\u0917 \u0938\u0924\u0930\u094d\u0915 \u0930\u0939\u0947\u0902\u0964",
    "Moderate": "\u092c\u091a\u094d\u091a\u0947, \u092c\u0941\u091c\u0941\u0930\u094d\u0917 \u0914\u0930 \u092e\u0930\u0940\u091c \u0932\u0902\u092c\u0947 \u0938\u092e\u092f \u0924\u0915 \u092c\u093e\u0939\u0930 \u0928 \u0930\u0939\u0947\u0902\u0964",
    "Poor": "\u092c\u093e\u0939\u0930 \u0935\u094d\u092f\u093e\u092f\u093e\u092e \u0928 \u0915\u0930\u0947\u0902\u0964 \u0918\u0930 \u0915\u0947 \u0905\u0902\u0926\u0930 \u0930\u0939\u0947\u0902\u0964",
    "Very Poor": "\u092c\u093e\u0939\u0930 \u0928 \u091c\u093e\u090f\u0902\u0964 \u0938\u094d\u0915\u0942\u0932 \u092e\u0947\u0902 \u092c\u093e\u0939\u0930\u0940 \u0917\u0924\u093f\u0935\u093f\u0927\u093f\u092f\u093e\u0902 \u0938\u0940\u092e\u093f\u0924 \u0915\u0930\u0947\u0902\u0964",
    "Severe": "\u0915\u0947\u0935\u0932 \u091c\u0930\u0942\u0930\u0940 \u0939\u094b\u0928\u0947 \u092a\u0930 \u0939\u0940 \u092c\u093e\u0939\u0930 \u091c\u093e\u090f\u0902, N95 \u092e\u093e\u0938\u094d\u0915 \u092a\u0939\u0928\u0947\u0902\u0964",
}


def severity_band(aqi: float) -> Dict:
    for lo, hi, label, advice in SEVERITY_BANDS:
        if lo <= aqi <= hi:
            return {"category": label, "advice": advice}
    return {"category": "Severe", "advice": SEVERITY_BANDS[-1][3]}


def generate_advisory(ward: str, current_aqi: float, tomorrow_aqi: float,
                       language: str = "English") -> Dict:
    band = severity_band(current_aqi)
    template = LANGUAGE_TEMPLATES.get(language, LANGUAGE_TEMPLATES["English"])

    text = "\n".join([
        template["header"].format(ward=ward),
        template["now"].format(aqi=round(current_aqi), category=band["category"]),
        template["tomorrow"].format(tomorrow_aqi=round(tomorrow_aqi)),
        template["advice"].format(advice=band["advice"], advice_hi=ADVICE_HINDI.get(band["category"], "")),
    ])

    return {
        "ward": ward, "language": language, "severity": band["category"],
        "current_aqi": round(current_aqi), "tomorrow_aqi": round(tomorrow_aqi),
        "text": text,
    }


def vulnerable_groups_flag(aqi: float) -> List[str]:
    """Which population groups should be flagged at this AQI level."""
    if aqi <= 100:
        return []
    if aqi <= 200:
        return ["children", "elderly", "asthma/heart patients"]
    return ["children", "elderly", "asthma/heart patients", "outdoor workers", "pregnant women"]


if __name__ == "__main__":
    out_en = generate_advisory("Rohini", 312, 245, "English")
    out_hi = generate_advisory("Rohini", 312, 245, "Hindi")
    print(out_en["text"])
    print("---")
    print(out_hi["text"])
    print("---")
    print("Vulnerable groups flagged:", vulnerable_groups_flag(312))
    assert out_en["severity"] == "Very Poor"
    print("advisory.py self-test passed.")
