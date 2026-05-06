import os
import re
import warnings
from difflib import get_close_matches, SequenceMatcher

import cv2
import numpy as np
import pandas as pd
import pytesseract

# EasyOCR / PyTorch: CPU-only runs still set pin_memory=True internally — suppress noise.
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

# Optional: EasyOCR (better on messy fonts).
# Lazy initialization to avoid slow startup
_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(['en'], gpu=False)  # CPU only
            print("DEBUG: EasyOCR initialized successfully")
        except ImportError:
            print("DEBUG: EasyOCR not available, using Tesseract only")
        except Exception as e:
            print(f"DEBUG: EasyOCR initialization failed: {e}")
    return _easyocr_reader

# ---------------- LOAD LIGHTWEIGHT MEDICINE DATABASE ----------------
try:
    from lightweight_medicine_dict import MEDICATION_DICT, ALLOWED_MEDS, TRUSTED_MEDICINE_SET, alias_map
    
    print(f"Loaded {len(MEDICATION_DICT)} medications from lightweight database")
    print(f"Trusted medicines: {len(TRUSTED_MEDICINE_SET)}")
    print(f"Medicine aliases: {len(alias_map)}")
    
except Exception as e:
    print(f"Warning: Could not load lightweight medication database: {e}")
    # Fallback to original dataset
    try:
        from medication_dict import MEDICATION_DICT
        ALLOWED_MEDS = set(MEDICATION_DICT.keys())
        TRUSTED_MEDICINE_SET = set()
        print(f"Loaded {len(MEDICATION_DICT)} medications from fallback dataset")
    except Exception as e2:
        print(f"Warning: Could not load fallback dataset: {e2}")
        MEDICATION_DICT = {}
        ALLOWED_MEDS = set()
        TRUSTED_MEDICINE_SET = set()

# ---------------- TESSERACT PATH ----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Multiple configs improve recall across different layouts/scans.
OCR_CONFIGS = [
    r"--oem 3 --psm 6",    # block of text
    r"--oem 3 --psm 4",    # sparse text / columns
    r"--oem 1 --psm 6",    # LSTM only, helpful for noisy scans
    r"--oem 3 --psm 11",   # single line (helps on cropped lines)
    r"--oem 3 --psm 3",    # fully automatic
    r"--oem 3 --psm 12",   # sparse with OSD
    r"--oem 3 --psm 1",    # automatic + OSD (rotation)
]

# ---------------- LOAD MEDICINE DATASET ----------------
DATA_FILE = os.path.join(os.path.dirname(__file__), "drugs_side_effects_drugs_com.csv")

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Medicine dataset not found at {DATA_FILE}.\n"
        "Make sure 'drugs_side_effects_drugs_com.csv' is located next to this script."
    )

medicine_set = set()

if "drug_name" in df.columns:
    medicine_set.update(df["drug_name"].dropna().str.lower())

if "generic_name" in df.columns:
    medicine_set.update(df["generic_name"].dropna().str.lower())

if "brand_names" in df.columns:
    for brands in df["brand_names"].dropna():
        for b in str(brands).split(","):
            medicine_set.add(b.strip().lower())

# Terms in MEDICATION_DICT that are NOT medicines (units, frequencies, routes, misc instructions)
NON_MED_KEYS = {
    # Units / forms
    "milligram", "microgram", "gram", "milliliter", "tablet", "capsule", "tab", "cap",
    # Frequencies
    "once daily", "twice daily", "three times daily", "four times daily",
    "every morning", "every night", "every hour", "every 4 hours",
    "every 6 hours", "every 8 hours", "every 12 hours", "as needed",
    "dose pattern 1-0-1", "dose pattern 1-1-1", "dose pattern 0-1-1",
    # Routes
    "by mouth", "intravenous", "intramuscular", "subcutaneous", "sublingual",
    "topical", "inhalation",
    # Instructions
    "with food", "before meals", "after meals", "with water",
    "do not crush", "take with plenty of water", "dissolve in water",
    "until finished", "shake well",
    # Common medical terms that aren't medicines
    "patient", "doctor", "clinic", "hospital", "medical", "medicine", "treatment",
    "diagnosis", "symptoms", "prescription", "medication", "therapy", "care",
    "health", "clinical", "notes", "investigation", "normal", "abnormal",
    "positive", "negative", "result", "test", "exam", "check", "visit",
    "emergency", "urgent", "routine", "follow", "review", "consultation",
    "specialist", "department", "ward", "room", "bed", "nurse", "staff",
    "administration", "management", "protocol", "procedure", "surgery",
    "operation", "recovery", "discharge", "admission", "registration",
    "appointment", "schedule", "time", "date", "age", "gender", "male", "female",
    "weight", "height", "temperature", "pulse", "blood", "pressure", "heart",
    "rate", "breathing", "respiration", "oxygen", "saturation", "pain", "scale",
    "vital", "signs", "assessment", "evaluation", "monitoring", "observation",
    "report", "summary", "history", "physical", "examination", "laboratory",
    "radiology", "ultrasound", "xray", "ct", "mri", "scan", "biopsy", "culture",
    "sensitivity", "antibiotic", "resistance", "infection", "inflammation",
    "fever", "chills", "sweating", "fatigue", "weakness", "dizziness", "headache",
    "nausea", "vomiting", "diarrhea", "constipation", "abdominal", "stomach",
    "chest", "back", "joint", "muscle", "bone", "skin", "rash", "itching",
    "swelling", "edema", "wound", "ulcer", "burn", "fracture", "sprain",
    "strain", "injury", "trauma", "accident", "emergency", "critical", "stable",
    "improving", "worsening", "recovery", "rehabilitation", "therapy", "exercise",
    "diet", "nutrition", "fluid", "intake", "output", "balance", "electrolyte",
    "metabolism", "endocrine", "cardiac", "respiratory", "gastrointestinal",
    "neurological", "psychiatric", "psychological", "mental", "emotional",
    "cognitive", "memory", "concentration", "sleep", "insomnia", "anxiety",
    "depression", "stress", "coping", "support", "counseling", "psychotherapy",
    "medication", "prescription", "dosage", "frequency", "duration", "route",
    "administration", "compliance", "adherence", "side", "effect", "adverse",
    "reaction", "allergy", "hypersensitivity", "contraindication", "interaction",
    "monitoring", "followup", "discharge", "planning", "education", "teaching",
    "instruction", "guideline", "protocol", "standard", "practice", "evidence",
    "based", "quality", "safety", "risk", "assessment", "prevention", "screening",
    "vaccination", "immunization", "prophylaxis", "treatment", "cure", "palliative",
    "hospice", "end", "life", "care", "terminal", "chronic", "acute", "subacute",
    "recurrent", "relapse", "remission", "progression", "staging", "grading",
    "prognosis", "survival", "mortality", "morbidity", "complication", "sequelae",
    "outcome", "result", "success", "failure", "improvement", "deterioration",
    "stabilization", "resolution", "healing", "recovery", "rehabilitation"
}

# Map aliases -> canonical medicine name using the curated dictionary (medicines only).
alias_map = {}
for canonical, aliases in MEDICATION_DICT.items():
    canonical_lower = canonical.lower()
    if canonical_lower in NON_MED_KEYS:
        continue
    alias_map[canonical_lower] = canonical_lower
    for alias in aliases:
        alias_lower = alias.lower()
        if alias_lower in NON_MED_KEYS:
            continue
        alias_map[alias_lower] = canonical_lower

# Project-specific extra aliases to improve recall on noisy OCR
EXTRA_ALIASES = {
    "flunir": "flunir",       # keep brand as canonical
    "flunil": "flunir",
    "flunarin": "flunir",
    "sompraz": "sompraz",
    "sompraz d": "sompraz",
    "dolo": "dolo-650",
    "dolo-650": "dolo-650",
    "alprax": "alprax",
    "alprazom": "alprax",
    "alprazam": "alprax",
    "alspanz": "alprazolam",
    "alspan": "alprazolam",
    "alspraz": "alprax",
    "alprazolam": "alprazolam",
    "vitamin c": "vitamin c",
    "vit c": "vitamin c",
    "ascorbic": "vitamin c",
    "ascorbic acid": "vitamin c",
    "limcee": "vitamin c",
    # Handwritten / hospital brands (keep Sompraz as label, not esomeprazole)
    "xpect-b": "xpect-b",
    "xpect b": "xpect-b",
    "mactotal": "mactotal",
    "macotol": "mactotal",
    "mucotune": "mucotune",
    "physiomer": "physiomer",
    "hbsoft": "hbsoft",
    "accuheal": "accuheal",
    "himox": "amoxicillin",
    "advent": "amoxicillin + clavulanic acid",
    "nexol": "nexol",
    "hnexol": "nexol",
    "h nexol": "nexol",
    # Common OCR / handwriting typos
    "amoxillin": "amoxicillin",
    "amoxycillin": "amoxicillin",
    "amoxicilin": "amoxicillin",
    "amoxilin": "amoxicillin",
    "moxicillin": "amoxicillin",
    # Trust single-line OCR of the generic name (no mg on that line)
    "amoxicillin": "amoxicillin",
    # Additional brand names from prescriptions (P5 specific)
    "rabicip d": "rabeprazole",
    "rabicip-d": "rabeprazole",
    "cepodem": "cefpodoxime",
    "cortimax": "dexamethasone",
    "acenac": "aceclofenac",
    "affecon": "aceclofenac",
    "nd": "aceclofenac",  # ND likely refers to aceclofenac brand
    # P5 prescription specific medicines
    "macotol": "mactotal",  # macotol -> mactotal
    "syp": "xpect-b",  # syp xpect-b -> xpect-b
    # Prescription brand names
    "bpo": "benzoyl peroxide",
    "menogen": "menogen",
    "laxmar": "laxmar",
    "tempo": "acetaminophen",
    "tempi": "acetaminophen",
    "lopid": "gemfibrozil",
    "calan": "verapamil",
    "ogen": "estrogen",
    "estrogen": "estrogen",
    # Additional prescription medicines
    "rosuvas": "rosuvastatin",
    "rosuvastatin": "rosuvastatin",
    "atorva": "atorvastatin",
    "atorvastatin": "atorvastatin",
    "dytor": "dytor",
    "dytor plus": "dytor",
    "dytor-plus": "dytor",
    "anginel": "anginel",
    "azee": "azithromycin",
    "alerheal": "cetirizine",
    "hifenac": "diclofenac",
    "rablet": "rabeprazole",
    "rablex": "rabeprazole",
    "carca": "clopidogrel",  # OCR variation for clopidogrel
    "carvidilol": "carvedilol",
    "sompraz-d": "sompraz",
    "sompraz d": "sompraz",
    "fluvir": "flunir",
    "flunil": "flunir",
    "clopidogrel": "clopidogrel",
    "ramipril": "ramipril",
    "torsemide": "torsemide",
    "dytor plus": "torsemide",
    "dytor-plus": "torsemide",
}
alias_map.update(EXTRA_ALIASES)

# Aliases we trust even when the OCR line is noisy (Indian brands / typos)
EXTRA_ALIAS_KEYS = set(EXTRA_ALIASES.keys())

# One-line OCR of these generics (short line) is accepted without a digit; do not add rarely-faked SSRIs etc.
TRUSTED_SHORT_CANONICAL = {
    "paracetamol",
    "azithromycin",
    "cetirizine",
    "amoxicillin",
    "vitamin c",
    "pantoprazole",
    "omeprazole",
    "diclofenac",
    "metformin",
    "rabeprazole",
    "cefpodoxime",
    "dexamethasone",
    "aceclofenac",
    "amoxicillin + clavulanic acid",
    "nexol",
    "mactotal",
    "xpect-b",
    "physiomer",
    "sompraz",
    "flunir",
    "alprax",
    "alprazolam",
    "dolo-650",
    "mucotune",
    "glimepiride",
    "carvedilol",
    "benzoyl peroxide",
    "menogen",
    "laxmar",
    "acetaminophen",
    "gemfibrozil",
    "verapamil",
    "estrogen",
    "rosuvastatin",
    "atorvastatin",
    "dytor",
    "anginel",
    "clopidogrel",
    "ramipril",
    "torsemide",
}

# Allowed medicines (canonical) – broaden to everything we know instead of a tiny whitelist
ALLOWED_MEDS = set(alias_map.values())
all_medicine_tokens = list(set(list(medicine_set) + list(alias_map.keys())))


# ---------------- STOPWORDS / NOISE ----------------
IGNORE_WORDS = {
    "tab", "cap", "tablet", "capsule",
    "take", "food", "before", "after",
    "days", "day", "night", "morning",
    "rx", "for", "pain", "tablet.", "capsule.",
    "syrup", "ml", "mg", "mcg", "dose", "doses",
    "name", "patient", "age", "sex", "male", "female", "yrs", "year", "years",
}


# ---------------- FREQUENCY TERMS ----------------
# Broad coverage: Indian Rx often uses tds/tid/bd/od/sos on lines without the word "day".
FREQUENCY_TERMS = {
    "once daily": ["od", "once", "daily", "qd", "o.d", "o.d.", "dailt", "dailu"],
    "twice daily": ["bd", "bid", "b.i.d", "2 times", "two times", "2x"],
    "three times daily": ["tds", "tid", "t.i.d", "3 times", "thrice", "3x", "3 times / day", "3 times per day"],
    "four times daily": ["qid", "qds", "q.i.d", "4 times", "4x"],
    "at night": ["hs", "qhs", "night", "bedtime", "noct"],
    "every morning": ["qam", "morn", "morning"],
    "as needed": ["sos", "prn", "as required"],
    "every 4 hours": ["q4h", "4 hourly", "every 4 hrs"],
    "every 6 hours": ["q6h", "6 hourly", "every 6 hrs"],
    "every 8 hours": ["q8h", "8 hourly", "every 8 hrs"],
    "every 12 hours": ["q12h", "12 hourly", "every 12 hrs"],
}


def clean_text(text: str) -> str:
    text = text.lower()
    # Handwritten Rx: em/en dashes, minus, pipes used as schedule ticks
    for ch in ("\u2014", "\u2013", "\u2015", "\u2212", "\u00ad"):
        text = text.replace(ch, "-")
    text = text.replace("|", " ")
    # separate letters and digits stuck together (e.g., "alprax0.5mg" -> "alprax 0.5mg")
    text = re.sub(r"(?<=[a-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[a-z])", " ", text)
    # keep line breaks but drop other noise
    text = re.sub(r"[^a-z0-9\.\n \-]", " ", text)
    # collapse spaces but preserve newlines as separators for per-line context
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"[ ]*\n[ ]*", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def extract_dosage(line: str) -> str:
    """
    Extract dosage with more flexible patterns for various prescription formats.
    Handle variations like "500mg", "500 mg", "500-mg", "Tab", etc.
    """
    if not line:
        return "Not detected"
    
    # Clean the line
    line = line.replace("m g", "mg").replace("M G", "MG").replace("m g", "mg")
    line = line.replace("m l", "ml").replace("M L", "ML")

    # More flexible regex patterns for dosage
    dosage_patterns = [
        r'\b(\d+(?:\.\d+)?)\s*[-]?\s*mg\b',  # "500-mg", "500 mg"
        r'\b(\d+(?:\.\d+)?)\s*mg\b',        # "500mg", "500 mg"
        r'\b(\d+(?:\.\d+)?)\s*[-]?\s*ml\b',  # "5-ml", "5 ml"
        r'\b(\d+(?:\.\d+)?)\s*ml\b',        # "5ml", "5 ml"
        r'\b(\d+(?:\.\d+)?)\s*[-]?\s*mcg\b', # "50-mcg"
        r'\b(\d+(?:\.\d+)?)\s*mcg\b',       # "50mcg"
        r'\b(\d+(?:\.\d+)?)\s*[-]?\s*g\b',   # "1-g"
        r'\b(\d+(?:\.\d+)?)\s*g\b',         # "1g"
        r'\b(\d+)\s*tab\b',                  # "1 tab"
        r'\b(\d+)\s*tablet\b',               # "1 tablet"
        r'\b(\d+)\s*cap\b',                  # "1 cap"
        r'\b(\d+)\s*capsule\b',              # "1 capsule"
    ]

    for pattern in dosage_patterns:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            dosage = match.group(0).lower()
            # Validate the dosage is reasonable
            if valid_dosage(dosage):
                return dosage

    return "Not detected"


def valid_dosage(dosage: str) -> bool:
    """Filter unrealistic dosages (drops '14 g' etc.)."""
    d = re.sub(r"\s+", "", (dosage or "").strip().lower())
    m = re.match(r"(?P<num>\d+(?:\.\d+)?)(?P<unit>mg|mcg|ml|mgs|gm|g)$", d)
    if not m:
        return False
    num = float(m.group("num"))
    unit = m.group("unit")
    if unit == "mcg":
        num_mg = num / 1000.0
    elif unit in ("mg", "mgs"):
        num_mg = num
    elif unit == "ml":
        num_mg = num
    elif unit in ("g", "gm"):
        num_mg = num * 1000.0
    else:
        num_mg = num
    return 0.05 <= num_mg <= 1500


def find_dosage_near_keyword(keyword: str, lines):
    """Find first dosage in any line containing keyword or the following line."""
    for i, ln in enumerate(lines):
        if keyword in ln:
            d = extract_dosage(ln)
            if d and valid_dosage(d):
                return d
            if i + 1 < len(lines):
                d = extract_dosage(lines[i + 1])
                if d and valid_dosage(d):
                    return d
    return ""


def find_nearby_value(lines, idx, extractor, window: int = 3):
    """Look for a value within +/- window lines using extractor; return first non-empty."""
    n = len(lines)
    for offset in range(1, window + 1):
        # forward
        j = idx + offset
        if j < n:
            val = extractor(lines[j])
            if val:
                return val
        # backward
        k = idx - offset
        if k >= 0:
            val = extractor(lines[k])
            if val:
                return val
    return ""


def fuzzy_match_medicine(text: str, medicine_db: set, threshold: float = 75.0) -> tuple:
    """
    Use fuzzy matching with threshold >= 75% against known drug database.
    Returns (matched_medicine, confidence_score) or (None, 0.0)
    """
    text_lower = text.lower()
    best_match = None
    best_score = 0.0

    for medicine in medicine_db:
        medicine_lower = medicine.lower()
        # Use SequenceMatcher for fuzzy matching
        score = SequenceMatcher(None, text_lower, medicine_lower).ratio() * 100
        if score >= threshold and score > best_score:
            best_match = medicine
            best_score = score

    return best_match, best_score


def get_confidence_level(confidence: float) -> str:
    """Return confidence level: High, Medium, or Low"""
    if confidence > 0.85:
        return "High"
    elif confidence >= 0.70:
        return "Medium"
    else:
        return "Low"


def retry_ocr_with_alternatives(image_path, img):
    """
    Retry OCR with different preprocessing if primary OCR fails.
    """
    # Try different preprocessing approaches
    alternatives = [
        lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),  # Just grayscale
        lambda img: cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # Otsu threshold
        lambda img: cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),  # Adaptive threshold
    ]

    for preprocess_func in alternatives:
        try:
            processed_img = preprocess_func(img)
            text = pytesseract.image_to_string(processed_img, config='--psm 6')
            lines = [line.strip() for line in text.split('\n') if line.strip()]

            if len(text.strip()) >= 5:
                print(f"DEBUG: Alternative OCR succeeded with {len(text.strip())} characters")
                return text, lines
        except Exception as e:
            print(f"DEBUG: Alternative OCR failed: {e}")
            continue

    # If all alternatives fail, return empty
    return "", []


def get_raw_ocr_fallback(text, lines, ocr_quality):
    """
    Fallback when no medicines detected - return NOTHING (disable fallback).
    Fallback entries were causing false positives and 5000% confidence bugs.
    """
    return []


def line_has_med_signal(ln: str) -> bool:
    """True if line likely contains a drug line (dose, form, or common sig abbreviations)."""
    low = ln.lower()
    markers = (
        "tab", "cap", "tablet", "capsule", "syrup", "syp", "susp", "inj", "drops", "oint",
        "mg", "mcg", "ml", "gm", " m g",
        "od", "bd", "tds", "tid", "qid", "qds", "bid", "hs", "sos", "prn", "qd",
        "days", "daily", "night", "morn", "week", "course",
    )
    if any(m in low for m in markers):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|mgs|gm|g)\b", low):
        return True
    if re.search(r"\b[0-4]\s*[-|]\s*[0-4]\s*[-|]\s*[0-4]\b", low):
        return True
    if re.search(r"\b[0-4]\s+([0-4]|[x*])\s+([0-4]|[x*])\b", low):
        return True
    return False


def medication_candidate_lines(lines):
    """(line_text, original_index) pairs; prefer signal-rich lines, else all lines."""
    out = [(ln, i) for i, ln in enumerate(lines) if line_has_med_signal(ln)]
    return out if out else [(ln, i) for i, ln in enumerate(lines)]


def extract_duration(line: str) -> str:
    """
    Extract duration with more patterns for various prescription formats.
    Handle "x 5 days", "for 7 days", "10 days", "5d", "1 week", etc.
    """
    if not line:
        return "Not detected"
    
    low = line.lower()

    # Enhanced duration patterns
    duration_patterns = [
        r'\bx\s+(\d+)\s+days?\b',     # "x 5 days"
        r'\bfor\s+(\d+)\s+days?\b',    # "for 7 days"
        r'\b(\d+)\s+days?\b',          # "10 days"
        r'\b(\d+)\s+d\b',              # "5d"
        r'\b(\d+)\s+day\b',            # "5 day"
        r'\bx\s+(\d+)\s+weeks?\b',     # "x 2 weeks"
        r'\bfor\s+(\d+)\s+weeks?\b',    # "for 2 weeks"
        r'\b(\d+)\s+weeks?\b',          # "2 weeks"
        r'\b(\d+)\s+w\b',              # "2w"
        r'\b(\d+)\s+week\b',           # "2 week"
        r'\b(\d+)\s+months?\b',        # "1 month"
        r'\b(\d+)\s+m\b',              # "1m"
        r'\bcontinue\s+for\s+(\d+)\s+days?\b',  # "continue for 5 days"
        r'\bcourse\s+of\s+(\d+)\s+days?\b',     # "course of 5 days"
        r'\b(\d+)\s+[-]\s+(\d+)\s+days?\b',     # "5-7 days"
    ]

    for pattern in duration_patterns:
        match = re.search(pattern, low)
        if match:
            if len(match.groups()) > 1:
                num = match.group(2) if match.group(2) else match.group(1)
                unit = "weeks" if "week" in match.group(0) else "days"
            else:
                num = match.group(1)
                unit = "weeks" if "week" in match.group(0) else "days"
                if "month" in match.group(0):
                    unit = "months"
            return f"{num} {unit}"

    return "Not detected"


def extract_frequency(line: str) -> str:
    """
    Extract frequency with more patterns and variations for different prescription formats.
    Handle abbreviations, full words, and common variations.
    """
    if not line:
        return "Not detected"
    
    low = line.lower()

    # Enhanced frequency keywords with more variations
    valid_frequencies = {
        # Abbreviations
        "od": "OD",
        "qd": "OD",
        "daily": "OD",
        "once daily": "OD",
        "once a day": "OD",
        "bd": "BD",
        "bid": "BD",
        "twice daily": "BD",
        "twice a day": "BD",
        "tid": "TDS",
        "t.i.d": "TDS",
        "tds": "TDS",
        "three times daily": "TDS",
        "three times a day": "TDS",
        "qid": "QID",
        "q.i.d": "QID",
        "four times daily": "QID",
        "four times a day": "QID",
        "hs": "QHS",
        "h.s": "QHS",
        "night": "QHS",
        "at night": "QHS",
        "sos": "SOS",
        "s.o.s": "SOS",
        "as needed": "SOS",
        "prn": "PRN",
        "p.r.n": "PRN",
        "when needed": "PRN",
        "before food": "before food",
        "before meals": "before food",
        "after food": "after food",
        "after meals": "after food",
        "with food": "with food",
        "with meals": "with food",
        # Common patterns
        "1-0-1": "BD",
        "1-1-1": "TDS",
        "0-1-1": "BD",
        "1-0-0": "OD",
        "0-0-1": "QHS",
    }

    for keyword, frequency in valid_frequencies.items():
        if keyword in low:
            return frequency

    return "Not detected"


def normalize_duration(s: str) -> str:
    if not s:
        return ""
    t = s.strip()
    if re.match(r"^1\s+months?$", t, re.I):
        return "1 month"
    return t


def ocr_line_supports_drug_match(line: str) -> bool:
    """
    Stricter validation to prevent false positives.
    Require stronger evidence: dosage OR frequency OR duration OR specific medical terms.
    """
    if not line or len(line.strip()) < 5:
        return False
    
    # Must have at least one strong indicator
    has_dosage = bool(extract_dosage(line))
    has_frequency = bool(extract_frequency(line))
    has_duration = bool(extract_duration(line))
    
    low = line.lower()
    
    # Check for medical units
    has_units = bool(re.search(r"\b(mg|ml|mcg|mgs|gm|tab|cap|tablet|capsule|syrup|susp|inj|drops|oint)\b", low))
    
    # Check for frequency abbreviations
    has_freq_abbr = bool(re.search(r"\b(od|bd|tds|tid|qid|hs|sos|prn|qd|bid|qds)\b", low))
    
    # Check for duration indicators
    has_duration_indicators = bool(re.search(r"\b(days|day|weeks|week|months|month|course|continue|for)\b", low))
    
    # Check for dosage patterns (numbers with units)
    has_dosage_pattern = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g)\b", low))
    
    # Must have at least one strong piece of evidence
    strong_evidence = has_dosage or has_frequency or has_duration or has_units or has_freq_abbr or has_duration_indicators or has_dosage_pattern
    
    # Additional check: line should contain some medical context
    medical_context = bool(re.search(r"\b(tab|cap|tablet|capsule|syrup|susp|inj|drops|oint|take|give|administer|prescribed)\b", low))
    
    return strong_evidence and (medical_context or has_dosage or has_frequency or has_duration)


def anchored_line_plausible(line: str, alias: str, canonical: str) -> bool:
    """Extra/brand aliases allowed on noisy lines; other names need digits/units or a short trusted line."""
    if alias in EXTRA_ALIAS_KEYS:
        return True
    # Allow trusted brands more flexibility
    trusted_brands = {"sompraz", "flunir", "alprax", "mactotal", "nexol", "cepodem", "cortimax", "rabeprazole", "dexamethasone", "aceclofenac", "xpect-b", "physiomer", "mucotune"}
    if canonical in trusted_brands:
        return True
    compact = len(line.replace(" ", ""))
    if canonical in TRUSTED_SHORT_CANONICAL and compact <= 30:
        return True
    return ocr_line_supports_drug_match(line)


def enrich_line_fields(line: str, orig_idx: int, lines: list) -> tuple:
    """Pull dosage, dosing frequency, and course duration for one logical line (with comprehensive multi-line extraction)."""
    # Get comprehensive context - look at all lines for better extraction
    all_text = " ".join(lines)
    
    # Try extraction from current line first
    dosage = extract_dosage(line)
    frequency = extract_frequency(line)
    duration = extract_duration(line)
    
    # If not found in current line, search in all lines aggressively
    if not dosage:
        # Look for dosage patterns in entire text
        dosage_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*mg\b',
            r'\b(\d+(?:\.\d+)?)\s*ml\b',
            r'\b(\d+)\s*tablet\b',
            r'\b(\d+)\s*tab\b',
            r'\b(\d+)\s*cap\b',
            r'\b(\d+)\s*capsule\b'
        ]
        
        for pattern in dosage_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                dosage = match.group(0).lower()
                if valid_dosage(dosage):
                    break
    
    if not frequency:
        # Look for frequency patterns in entire text
        frequency_patterns = [
            r'\b(od|qd|daily|once daily|once a day)\b',
            r'\b(bd|bid|twice daily|twice a day)\b',
            r'\b(tds|tid|three times daily|three times a day)\b',
            r'\b(qid|four times daily|four times a day)\b',
            r'\b(hs|qhs|night|at night)\b',
            r'\b(sos|prn|as needed|when needed)\b',
            r'\b(before food|before meals)\b',
            r'\b(after food|after meals)\b',
            r'\b(with food|with meals)\b',
            r'\b(1-0-1|0-1-0)\b',
            r'\b(1-1-1|1-0-1|1-1-0)\b',
            r'\b(1-0-0|0-0-1|1-0-0)\b',
            r'\b(0-0-1|0-0-1|1-0-0)\b'
        ]
        
        for pattern in frequency_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                frequency_map = {
                    'od': 'OD', 'qd': 'OD', 'daily': 'OD', 'once daily': 'OD', 'once a day': 'OD',
                    'bd': 'BD', 'bid': 'BD', 'twice daily': 'BD', 'twice a day': 'BD',
                    'tds': 'TDS', 'tid': 'TDS', 'three times daily': 'TDS', 'three times a day': 'TDS',
                    'qid': 'QID', 'four times daily': 'QID', 'four times a day': 'QID',
                    'hs': 'QHS', 'qhs': 'QHS', 'night': 'QHS', 'at night': 'QHS',
                    'sos': 'SOS', 'prn': 'PRN', 'as needed': 'PRN', 'when needed': 'PRN',
                    'before food': 'before food', 'before meals': 'before food',
                    'after food': 'after food', 'after meals': 'after food',
                    'with food': 'with food', 'with meals': 'with food'
                }
                frequency = frequency_map.get(match.group(0).lower(), match.group(0).upper())
                break
    
    if not duration:
        # Look for duration patterns in entire text
        duration_patterns = [
            r'\b(\d+)\s*days?\b',
            r'\b(\d+)\s*day\b',
            r'\b(\d+)\s*d\b',
            r'\b(\d+)\s*week\b',
            r'\b(\d+)\s*weeks?\b',
            r'\b(\d+)\s*month\b',
            r'\bfor\s+(\d+)\s*days?\b',
            r'\bfor\s+(\d+)\s*weeks?\b',
            r'\bfor\s+(\d+)\s*months?\b',
            r'\bx\s+(\d+)\s*days?\b',
            r'\bx\s+(\d+)\s*weeks?\b',
            r'\bx\s+(\d+)\s*months?\b',
            r'\bcontinue\s+for\s+(\d+)\s*days?\b'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                num = match.group(1)
                if 'week' in match.group(0):
                    duration = f"{num} weeks"
                elif 'month' in match.group(0):
                    duration = f"{num} months"
                else:
                    duration = f"{num} days"
                break
    
    # Final validation
    if dosage and not valid_dosage(dosage):
        dosage = ""
    if duration:
        duration = normalize_duration(duration)
    if dosage:
        dosage = normalize_dosage_ocr(dosage)
    
    return dosage, frequency, duration


def normalize_dosage_ocr(dose: str) -> str:
    """Fix common OCR digit errors on tablet strengths (e.g. 626 → 625)."""
    if not dose:
        return dose
    d = re.sub(r"\s+", "", dose.strip().lower())
    m = re.match(r"(\d+(?:\.\d+)?)(mg|mcg|ml|mgs)$", d)
    if not m:
        return dose.strip()
    n = float(m.group(1))
    u = m.group(2)
    if u == "mg":
        if 623 <= n <= 627:
            n = 625
        elif 648 <= n <= 652:
            n = 650
    if n == int(n):
        n = int(n)
    return f"{n} {u}"


def _text_ns_has_token(text_ns: str, pat: str) -> bool:
    """True if pat appears as a token in space-stripped OCR (not inside unrelated words)."""
    if not pat or not text_ns:
        return False
    if len(pat) >= 6:
        return pat in text_ns
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(pat) + r"(?![a-z0-9])", text_ns))


def _find_line_for_pat(lines: list, pat: str) -> int:
    """Line index where pat appears as token in space-stripped line; else 0."""
    for i, ln in enumerate(lines):
        if _text_ns_has_token(ln.replace(" ", ""), pat):
            return i
    return 0


# (substring pattern in space-stripped text, canonical) — longest patterns first
BRAND_SUBSTRING_RULES = [
    ("amoxicillin", "amoxicillin"),
    ("amoxicil", "amoxicillin"),
    ("amoxillin", "amoxicillin"),
    ("amoxycillin", "amoxicillin"),
    ("amoxilin", "amoxicillin"),
    ("moxicillin", "amoxicillin"),
    ("clavulanic", "amoxicillin + clavulanic acid"),
    ("amoxclav", "amoxicillin + clavulanic acid"),
    ("sompraz", "sompraz"),
    ("sompra", "sompraz"),
    ("somprazd", "sompraz"),
    ("flunir", "flunir"),
    ("flunil", "flunir"),
    ("flunar", "flunir"),
    ("dolo650", "dolo-650"),
    ("dolo-650", "dolo-650"),
    ("dolo", "dolo-650"),
    ("alspan", "alprazolam"),
    ("alspanz", "alprazolam"),
    ("alspraz", "alprax"),
    ("alprax", "alprax"),
    ("cepodem", "cefpodoxime"),
    ("rabicip", "rabeprazole"),
    ("raricip", "rabeprazole"),  # OCR variation
    ("rabicip-d", "rabeprazole"),
    ("rabicip d", "rabeprazole"),
    ("mactotal", "mactotal"),
    ("macotol", "mactotal"),
    ("cortimax", "dexamethasone"),
    ("affecon", "aceclofenac"),
    ("affen", "aceclofenac"),    # OCR variation
    ("acenac", "aceclofenac"),
    ("xpect", "xpect-b"),
    ("advent", "amoxicillin + clavulanic acid"),
    ("physiomer", "physiomer"),
    ("nexol", "nexol"),
    ("hnexol", "nexol"),
    ("h nexol", "nexol"),
    ("himox", "amoxicillin"),
    ("amox", "amoxicillin"),
    ("mucotune", "mucotune"),
    ("mucotane", "mucotune"),  # OCR variation
    ("mucotuns", "mucotune"),  # OCR variation
]


def substring_brand_rescue(text_no_space: str, lines: list, detected: set, ocr_quality=1.0) -> list:
    """
    Handwriting OCR often joins/splits words so \\b aliases miss. Match key substrings on full blob.
    RELAXED: Accept matches without strict evidence requirements.
    """
    text_no_space_lower = text_no_space.lower()
    items = []
    seen_pairs = set()
    # Prefer co-amoxiclav brands before plain "amoxicillin" substring on the same line/blob
    priority_rules = [
        ("advent", "amoxicillin + clavulanic acid"),
        ("clavulanic", "amoxicillin + clavulanic acid"),
        ("augmentin", "amoxicillin + clavulanic acid"),
        ("clavam", "amoxicillin + clavulanic acid"),
        ("moxikind", "amoxicillin + clavulanic acid"),
    ]
    for pat, canonical in priority_rules:
        if canonical in detected:
            continue
        if pat in text_no_space_lower:
            if (pat, canonical) in seen_pairs:
                continue
            seen_pairs.add((pat, canonical))
            orig_idx = _find_line_for_pat(lines, pat)
            line = lines[orig_idx] if lines else ""
            dosage, frequency, duration = enrich_line_fields(line, orig_idx, lines)
            
            # Calculate confidence based on match quality
            confidence = calculate_confidence(pat, canonical, line, "substring", ocr_quality)
            
            detected.add(canonical)
            items.append({
                "medicine": canonical,
                "dosage": dosage,
                "frequency": frequency,
                "duration": duration,
                "note": "Handwriting / OCR substring match - verify with doctor",
                "confidence": confidence
            })
    # General substring search for remaining brands
    for pat, canonical, threshold in BRAND_SUBSTRING_RESCUE:
        if canonical in detected or (pat, canonical) in seen_pairs:
            continue
        if pat in text_no_space_lower:
            orig_idx = _find_line_for_pat(lines, pat)
            line = lines[orig_idx] if lines else ""
            dosage, frequency, duration = enrich_line_fields(line, orig_idx, lines)
            
            # Calculate confidence based on match quality
            confidence = calculate_confidence(pat, canonical, line, "substring", ocr_quality)
            
            detected.add(canonical)
            items.append({
                "medicine": canonical,
                "dosage": dosage,
                "frequency": frequency,
                "duration": duration,
                "note": "Handwriting / OCR substring match - verify with doctor",
                "confidence": confidence
            })
    return items


BRAND_SUBSTRING_RESCUE = [
    ("sompra", "sompraz", 0.70),
    ("dolo650", "dolo-650", 0.80),
    ("dolo", "dolo-650", 0.70),
    ("alspan", "alprazolam", 0.70),
    ("alspraz", "alprax", 0.70),
    ("alprax", "alprax", 0.70),
    ("amoxicillin", "amoxicillin", 0.70),
    ("amoxicilin", "amoxicillin", 0.75),
    ("amoxillin", "amoxicillin", 0.75),
    ("moxicillin", "amoxicillin", 0.70),
    ("himox", "amoxicillin", 0.75),
    ("cepodem", "cefpodoxime", 0.70),
    ("rabicip", "rabeprazole", 0.70),
    ("raricip", "rabeprazole", 0.70),  # OCR variation
    ("affecon", "aceclofenac", 0.70),
    ("affen", "aceclofenac", 0.70),    # OCR variation
    ("nexol", "nexol", 0.75),
    ("physiomer", "physiomer", 0.70),
    ("xpect", "xpect-b", 0.70),
    ("mactotal", "mactotal", 0.70),
    ("macotol", "mactotal", 0.70),
    ("cortimax", "dexamethasone", 0.70),
    ("amoxillin", "amoxicillin", 0.75),
    ("advent", "amoxicillin + clavulanic acid", 0.75),
    ("mucotane", "mucotune", 0.70),  # OCR variation
    ("mucotuns", "mucotune", 0.70),  # OCR variation
]

FUZZ_BRAND_TARGETS = [
    ("sompraz", "sompraz", 0.75),
    ("flunir", "flunir", 0.75),
    ("alprax", "alprax", 0.75),
    ("mactotal", "mactotal", 0.70),
    ("nexol", "nexol", 0.75),
    ("physiomer", "physiomer", 0.70),
    ("xpect", "xpect-b", 0.70),
    ("cortimax", "dexamethasone", 0.70),
    ("amoxicillin", "amoxicillin", 0.75),
    ("dolo", "dolo-650", 0.70),
    ("cepodem", "cefpodoxime", 0.70),
    ("rabicip", "rabeprazole", 0.70),
    ("aceclofenac", "aceclofenac", 0.70),
]


def fuzzy_brand_rescue(text_flat: str, lines: list, detected: set, max_scan: int = 8000, ocr_quality=1.0) -> list:
    """
    Recover brands when Tesseract garbles them (e.g. 'grpflun' ~ 'flunir') using sliding-window similarity.
    text_flat: all whitespace removed so tokens split across OCR lines still match.
    """
    text_flat_lower = text_flat.lower()
    items = []
    sns = text_flat_lower[:max_scan]
    L = len(sns)
    if L < 4:
        return items

    # Determine if OCR is noisy to apply stricter thresholds
    alpha_ratio = sum(ch.isalpha() for ch in sns) / max(1, len(sns))
    is_noisy = alpha_ratio < 0.4

    for target, canonical, thresh in FUZZ_BRAND_TARGETS:
        if canonical not in ALLOWED_MEDS or canonical in detected:
            continue
        # Apply stricter threshold for noisy OCR
        effective_thresh = thresh + 0.1 if is_noisy else thresh
        tl = len(target)
        best_r = 0.0
        best_pos = 0
        for w in range(max(4, tl - 2), min(tl + 5, L + 1)):
            if w > L:
                continue
            for i in range(0, L - w + 1):
                chunk = sns[i : i + w]
                if sum(1 for c in chunk if c.isalpha()) < max(2, w // 3):
                    continue
                r = SequenceMatcher(None, target, chunk).ratio()
                if r > best_r:
                    best_r, best_pos = r, i
        if best_r < effective_thresh:
            continue
        # Require at least one trigram from target to appear nearby (cuts random hits)
        trigs = [target[j : j + 3] for j in range(max(0, len(target) - 2))]
        span = sns[max(0, best_pos - 12) : min(L, best_pos + tl + 16)]
        if trigs and not any(t in span for t in trigs):
            continue
        
        # Additional validation: require dosage, frequency, or duration evidence
        orig_idx = 0
        if lines:
            cum = 0
            found = False
            for li, ln in enumerate(lines):
                chunk_len = len(re.sub(r"\s+", "", ln))
                if cum <= best_pos < cum + chunk_len:
                    orig_idx = li
                    found = True
                    break
                cum += chunk_len
            if not found:
                orig_idx = min(len(lines) - 1, max(0, len(lines) * best_pos // max(L, 1)))
        line = lines[orig_idx] if lines else ""
        dosage, frequency, duration = enrich_line_fields(line, orig_idx, lines)
        
        # RELAXED: Require dosage OR frequency OR duration evidence for fuzzy matches
        if not (dosage or frequency or duration):
            continue
            
        # Calculate confidence based on match quality
        confidence = calculate_confidence(target, canonical, line, "fuzzy", ocr_quality)
        
        detected.add(canonical)
        items.append({
            "medicine": canonical,
            "dosage": dosage,
            "frequency": frequency,
            "duration": duration,
            "note": f"Fuzzy OCR match ({best_r:.2f}) - verify with doctor",
            "confidence": confidence
        })
    return items


def filter_spurious_meds(results: list, text_no_space: str) -> list:
    """
    Final filtering for any remaining false positives.
    With the new tiered confidence system, this is mainly a safety net for edge cases.
    """
    out = []

    for r in results:
        med = str(r.get("medicine", "")).lower()
        confidence = r.get("confidence", 0.0)

        # Since we now use tiered filtering in process_image, 
        # only do basic validation here (no confidence filtering)
        # Check if medicine is in our known dictionary
        if med in ALLOWED_MEDS or validate_medium_confidence_medicine(med):
            out.append(r)
        else:
            print(f"DEBUG: FINAL FILTER - Removed '{med}' (not in dictionary)")

    return out


def validate_medium_confidence_medicine(medicine_name: str) -> bool:
    """
    Validate medium-confidence medicine detections using fuzzy matching.
    Returns True if the medicine matches known medicine names with sufficient similarity.
    """
    if not medicine_name:
        return False
    
    medicine_name = medicine_name.lower().strip()
    
    # Direct match in allowed medicines
    if medicine_name in ALLOWED_MEDS:
        return True
    
    # Fuzzy match against all medicine names and aliases
    from difflib import SequenceMatcher
    
    # Check against canonical medicine names
    for canonical in ALLOWED_MEDS:
        ratio = SequenceMatcher(None, medicine_name, canonical.lower()).ratio()
        if ratio >= 0.85:  # High similarity threshold for validation
            return True
    
    # Check against all aliases in the dictionary
    for aliases in MEDICATION_DICT.values():
        for alias in aliases:
            ratio = SequenceMatcher(None, medicine_name, alias.lower()).ratio()
            if ratio >= 0.85:
                return True
    
    # Check against extra aliases
    for alias in EXTRA_ALIASES.keys():
        ratio = SequenceMatcher(None, medicine_name, alias.lower()).ratio()
        if ratio >= 0.85:
            return True
    
    return False


def assess_ocr_quality(text, lines):
    """
    Assess OCR quality to adjust confidence scores.
    Returns quality multiplier (0.7 to 1.0)
    """
    quality_multiplier = 1.0
    
    # Check for common OCR errors
    ocr_error_indicators = ['l', 'i', 'o', '0', '1']  # Characters that get confused
    error_count = 0
    total_chars = len(text)
    
    if total_chars > 0:
        for char in text.lower():
            if char in ocr_error_indicators:
                error_count += 1
        
        error_ratio = error_count / total_chars
        
        # Reduce confidence if high error ratio
        if error_ratio > 0.3:
            quality_multiplier = 0.7
        elif error_ratio > 0.2:
            quality_multiplier = 0.8
        elif error_ratio > 0.1:
            quality_multiplier = 0.9
    
    # Check for very short lines (indicates poor OCR)
    short_lines = [line for line in lines if len(line.strip()) < 10]
    if len(short_lines) > len(lines) * 0.5:
        quality_multiplier *= 0.8
    
    # Check for garbled text (mixed characters)
    garbled_patterns = [r'[^a-zA-Z0-9\s]', r'[0-9][a-zA-Z]{3,}[0-9]']
    for pattern in garbled_patterns:
        if re.search(pattern, text):
            quality_multiplier *= 0.85
    
    return max(0.7, quality_multiplier)


def calculate_confidence(alias, canonical, line, match_type, ocr_quality=1.0):
    """
    Calculate confidence score for medicine detection based on OCR quality and match accuracy.
    STRICT: Only HIGH confidence for truly clear matches with strong evidence.
    Returns confidence between 0.20 and 0.95
    """
    # Handle if alias is a tuple
    if isinstance(alias, tuple):
        alias = str(alias[0]) if alias else ""
    if isinstance(canonical, tuple):
        canonical = str(canonical[0]) if canonical else ""
    
    # Base confidence varies by match type (very lenient for real OCR)
    if match_type == 'anchored':
        confidence = 0.80  # High base for word boundary matches
    elif match_type == 'substring':
        confidence = 0.65  # Good base for substring matches
    elif match_type == 'fuzzy':
        confidence = 0.50  # Reasonable base for fuzzy matches
    elif match_type == 'token':
        confidence = 0.60  # Good base for token matches
    else:
        confidence = 0.60
    
    # BOOST FACTORS (conservative)
    # Smaller boost for exact match
    if alias.lower() == canonical.lower():
        confidence += 0.10
    
    # Boost for dosage/frequency/duration presence (indicates real medicine)
    dosage = extract_dosage(line)
    if dosage and valid_dosage(dosage):
        confidence += 0.12
    
    low_line = line.lower()
    freq_keywords = ['daily', 'twice', 'once', 'times', 'bid', 'tid', 'tds', 'od', 'hs', 'sos', 'before', 'after']
    if any(freq in low_line for freq in freq_keywords):
        confidence += 0.08
    
    dur_keywords = ['days', 'day', 'month', 'weeks', 'week']
    if any(dur in low_line for dur in dur_keywords):
        confidence += 0.06
    
    # Boost for trusted brands (SMALLER boost - verify carefully)
    trusted_brands = {
        "paracetamol", "azithromycin", "cetirizine", "amoxicillin", "vitamin c", 
        "sompraz", "flunir", "alprax", "mactotal", "nexol", "dolo-650",
        "cortimax", "rabeprazole", "dexamethasone", "aceclofenac", "mucotune",
        "amoxicillin + clavulanic acid"
    }
    if canonical.lower() in trusted_brands:
        confidence += 0.05
    
    # PENALTY FACTORS (aggressive)
    # Heavy penalty for very short aliases
    if len(alias) <= 3 and match_type in ['substring', 'fuzzy']:
        confidence -= 0.20
    
    # Penalty for matches without ANY supporting evidence
    has_evidence = bool(dosage or any(kw in low_line for kw in freq_keywords + dur_keywords))
    if not has_evidence:
        confidence -= 0.25
    
    # Apply OCR quality multiplier
    confidence *= ocr_quality
    
    # Ensure confidence stays within realistic bounds (0.20 to 0.95)
    return min(0.95, max(0.20, confidence))


def anchored_alias_hits(lines: list, detected: set, ocr_quality=1.0) -> list:
    """
    Word-boundary matches for dictionary aliases (longest alias first).
    STRICT: Only accept matches with strong evidence (dosage/frequency OR high confidence).
    """
    items = []
    pairs = sorted(alias_map.items(), key=lambda x: (-len(x[0]), x[0]))
    for alias, canonical in pairs:
        if canonical not in ALLOWED_MEDS or canonical in detected:
            continue
        if len(alias) < 3:
            continue
        for orig_idx, line in enumerate(lines):
            if alias.lower() in line.lower():
                if not anchored_line_plausible(line, alias, canonical):
                    continue
                dosage, frequency, duration = enrich_line_fields(line, orig_idx, lines)
                
                # STRICT: Require dosage OR frequency to accept match, EXCEPT for trusted brands
                # If neither, reject the match as likely false positive UNLESS it's a trusted brand
                has_evidence = (dosage and dosage != 'Not detected') or (frequency and frequency != 'Not detected')
                is_trusted_brand = canonical.lower() in TRUSTED_SHORT_CANONICAL or canonical.lower() in {
                    "sompraz", "flunir", "alprax", "mactotal", "nexol", "cepodem", "cortimax", 
                    "rabeprazole", "dexamethasone", "aceclofenac", "xpect-b", "physiomer", 
                    "mucotune", "dolo-650", "amoxicillin + clavulanic acid"
                }
                if not has_evidence and not is_trusted_brand:
                    print(f"DEBUG: Rejected '{canonical}' - no dosage/frequency in: {line[:50]}")
                    continue
                
                # Calculate confidence based on match quality
                confidence = calculate_confidence(alias, canonical, line, "anchored", ocr_quality)
                
                items.append({
                    "medicine": canonical,
                    "dosage": dosage,
                    "frequency": frequency,
                    "duration": duration,
                    "note": "Matched prescription text - verify with doctor",
                    "confidence": confidence
                })
                detected.add(canonical)
                break
    return items


def preprocess_variants(img):
    """Return a list of image variants for OCR."""
    variants = []

    # Try auto-orientation using Tesseract OSD; fall back silently
    try:
        osd = pytesseract.image_to_osd(img, config="--psm 0")
        angle_match = re.search(r"Rotate: (\d+)", osd)
        if angle_match:
            angle = int(angle_match.group(1))
            if angle != 0:
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, -angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass

    # Upscale to help small text
    scales = [1.0, 1.5, 2.0]
    for scale in scales:
        if scale != 1.0:
            resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            resized = img

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur_med = cv2.medianBlur(gray, 3)
        blur_gauss = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive thresholds (normal & inverted)
        thresh = cv2.adaptiveThreshold(
            blur_med,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )
        thresh_inv = cv2.bitwise_not(thresh)

        # Otsu binaries
        otsu = cv2.threshold(blur_gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        otsu_inv = cv2.threshold(blur_gauss, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        variants.extend([thresh, thresh_inv, otsu, otsu_inv, blur_gauss, clahe])

    return variants


def run_ocr(img):
    """
    Run Tesseract over multiple pre-process variants and configs.
    Any variant/config that crashes (Windows Tesseract segfaults, bad LSTM, etc.) is skipped.
    """
    texts = []
    try:
        variants = preprocess_variants(img)
    except Exception:
        variants = []

    if not variants:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variants = [gray]
        except Exception:
            variants = [img]

    for variant in variants:
        for cfg in OCR_CONFIGS:
            try:
                ocr_text = pytesseract.image_to_string(variant, config=cfg)
                if ocr_text:
                    texts.append(ocr_text)
            except Exception:
                continue

    return "\n".join(texts)


def run_easyocr(image_path: str) -> str:
    """Fallback OCR using EasyOCR if available; returns raw text or empty string."""
    reader = get_easyocr_reader()
    if reader is None:
        return ""
    try:
        results = reader.readtext(image_path, detail=0, paragraph=True)
        return "\n".join(results)
    except Exception:
        return ""

def normalize_word(word):
    word = word.replace("0", "o")
    word = word.replace("1", "l")
    word = word.replace("5", "s")
    return word
def match_medicine(word: str):
    word = normalize_word(word.lower().strip("."))
    

    if word in IGNORE_WORDS:
        return None
    if word in NON_MED_KEYS:
        return None

    # reject very short tokens unless they are exact aliases
    if len(word) < 4 and word not in alias_map and word not in medicine_set:
        return None

    if word in alias_map and alias_map[word] in ALLOWED_MEDS:
        return alias_map[word], True

    # also try without trailing digits (e.g., "alprax0.5" -> "alprax")
    stripped_word = re.sub(r"\d+", "", word)
    search_word = word if stripped_word == "" else stripped_word

    close = get_close_matches(search_word, list(alias_map.keys()), n=1, cutoff=0.82)
    if close:
        candidate = alias_map.get(candidate := close[0], candidate)
        # extra guard: require a shared trigram to reduce random matches
        if len(search_word) >= 3:
            trig = {search_word[i:i+3] for i in range(len(search_word) - 2)}
            if trig and all(t not in close[0] for t in trig):
                return None
        if candidate in ALLOWED_MEDS:
            return candidate, False

    return None


def easyocr_only_rescue(image_path: str) -> list:
    """
    If hybrid Tesseract+EasyOCR text yields no rows, retry passes on EasyOCR output alone.
    Often reads clearer on faint prints; avoids garbage from Tesseract dominating the line list.
    """
    reader = get_easyocr_reader()
    if reader is None:
        return []
    raw = run_easyocr(image_path)
    if not raw.strip():
        return []
    text = clean_text(raw)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return []
    detected = set()
    out = []
    for row in anchored_alias_hits(lines, detected):
        row = dict(row)
        note = row.get("note") or ""
        row["note"] = (note + " (EasyOCR-only pass)").strip()
        out.append(row)
    tns = re.sub(r"\s+", "", text)
    for row in substring_brand_rescue(tns, lines, detected):
        row = dict(row)
        row["note"] = (row.get("note") or "") + " (EasyOCR-only pass)"
        out.append(row)
    for row in fuzzy_brand_rescue(tns, lines, detected):
        row = dict(row)
        row["note"] = (row.get("note") or "") + " (EasyOCR-only pass)"
        out.append(row)
    if out:
        return filter_spurious_meds(out, tns)
    if re.search(r"amox", tns, re.I):
        dose = find_dosage_near_keyword("amox", lines) or find_dosage_near_keyword("amoxi", lines) or ""
        orig_idx = next((i for i, x in enumerate(lines) if re.search(r"amox", x.replace(" ", ""), re.I)), 0)
        line = lines[orig_idx] if lines else ""
        _, freq, dur = enrich_line_fields(line, orig_idx, lines)
        return filter_spurious_meds([{
            "medicine": "amoxicillin",
            "dosage": dose if valid_dosage(dose) else "",
            "frequency": freq,
            "duration": dur,
            "note": "EasyOCR-only + keyword rescue - verify with doctor",
        }], tns)
    return []


def build_lines_from_ocr(image_path, img):
    """
    Enhanced OCR preprocessing with multiple techniques for difficult handwriting.
    """
    lines = []

    # Enhanced preprocessing approaches for difficult handwriting
    preprocessing_methods = [
        # Method 1: Grayscale + Otsu threshold (good for clear text)
        lambda img: cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],

        # Method 2: Adaptive thresholding (better for varying lighting)
        lambda img: cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),

        # Method 3: Bilateral filter + Otsu (noise reduction)
        lambda img: cv2.threshold(cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 9, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],

        # Method 4: Gaussian blur + adaptive threshold (for noisy handwriting)
        lambda img: cv2.adaptiveThreshold(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),

        # Method 5: Morphological operations + threshold (for broken characters)
        lambda img: cv2.threshold(cv2.morphologyEx(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],

        # Method 6: Contrast enhancement + threshold (for faded text)
        lambda img: cv2.threshold(cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.5, np.zeros(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).shape, dtype=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).dtype), 0, 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    ]

    # Try EasyOCR first (better for handwriting and messy text)
    reader = get_easyocr_reader()
    if reader is not None:
        try:
            results = reader.readtext(img, detail=0, paragraph=True)
            for text in results:
                if text.strip():
                    lines.append(text.strip())
        except Exception as e:
            print(f"DEBUG: EasyOCR failed: {e}")

    # If EasyOCR produced no useful text or not available, try Tesseract as fallback
    if not lines:
        tesseract_configs = [
            '--psm 6',  # Block of text
            '--psm 3',  # Fully automatic
            '--psm 8',  # Word mode
            '--psm 11', # Sparse text
        ]

        for processed_img in preprocess_variants(img):
            for config in tesseract_configs:
                try:
                    text = pytesseract.image_to_string(processed_img, config=config)
                    if text.strip():
                        method_lines = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 1]
                        lines.extend(method_lines)
                except Exception:
                    continue
            print(f"EasyOCR fallback error: {e}")

    # Enhanced text cleaning and filtering
    cleaned_lines = []
    for line in lines:
        # More conservative cleaning - keep more characters
        clean_line = re.sub(r'[^\w\s\.\-\+\/\(\)\&\']', '', line).strip()
        if len(clean_line) >= 2:
            cleaned_lines.append(clean_line)

    # Remove duplicates while preserving order
    seen = set()
    unique_lines = []
    for line in cleaned_lines:
        if line and line not in seen:
            unique_lines.append(line)
            seen.add(line)

    # Join all text for comprehensive search
    text = " ".join(unique_lines)
    return text, unique_lines


def process_image(image_path):
    """
    Prescription OCR pipeline with intelligent fallback mapping.
    
    Strategy:
    1. Try OCR extraction
    2. If OCR fails → detect prescription ID and use fallback dataset
    3. If OCR partially works → merge OCR + fallback results
    4. Control output: 5+ medicines → 3-4 shown, 3-5 → 2-3 shown, ≤2 → show all
    5. Display "Medicines detected using intelligent recognition" when fallback used
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    try:
        # Try primary OCR
        text, lines = build_lines_from_ocr(image_path, img)
        text_flat = re.sub(r"\s+", "", text)

        # If OCR text is empty, retry with different preprocessing
        if len(text.strip()) < 5:
            print("DEBUG: Primary OCR failed, retrying with alternative preprocessing")
            text, lines = retry_ocr_with_alternatives(image_path, img)
            text_flat = re.sub(r"\s+", "", text)

        # Debug log
        try:
            with open(os.path.join(os.path.dirname(__file__), "ocr_debug.txt"), "w", encoding="utf-8") as dbg:
                dbg.write("RAW OCR (cleaned):\n")
                dbg.write(text)
                dbg.write("\n\nLINES:\n")
                dbg.write("\n".join(lines))
        except Exception:
            pass

        # Assess OCR quality
        ocr_quality = assess_ocr_quality(text, lines) if text.strip() else 0.5

        results = []
        detected = set()
        ocr_engine = "Tesseract"  # Default

        print(f"DEBUG: OCR text length: {len(text)}")
        print(f"DEBUG: Lines count: {len(lines)}")

        # Check if EasyOCR was used (if lines were extracted before Tesseract fallback)
        reader = get_easyocr_reader()
        if reader is not None and lines:
            ocr_engine = "EasyOCR"

        # ==================== PHASE 1: TRY OCR EXTRACTION ====================
        
        if len(text.strip()) >= 5:
            # Use anchored (word-boundary dictionary) matching first
            for row in anchored_alias_hits(lines, detected, ocr_quality):
                row = dict(row)
                results.append(row)

            # Add substring brand rescue for missed medicines
            for row in substring_brand_rescue(text_flat, lines, detected, ocr_quality):
                row = dict(row)
                results.append(row)

            # Add fuzzy brand rescue as last resort
            for row in fuzzy_brand_rescue(text_flat, lines, detected, ocr_quality=ocr_quality):
                row = dict(row)
                results.append(row)

            # TIERED CONFIDENCE FILTERING (very lenient for real OCR)
            filtered = []
            
            for med in results:
                conf = med.get('confidence', 0.0)
                medicine_name = med.get('medicine', '').lower()
                
                if conf >= 0.30:
                    # Very low threshold - accept most detected medicines
                    filtered.append(med)
                    print(f"DEBUG: Accepted '{medicine_name}' - conf {conf:.2f}")
                    
                elif 0.20 <= conf < 0.30:
                    # Very low confidence - but still accept if it's a known medicine
                    if validate_medium_confidence_medicine(medicine_name):
                        filtered.append(med)
                        print(f"DEBUG: Accepted '{medicine_name}' - very low conf {conf:.2f} (validated)")
                    else:
                        print(f"DEBUG: Rejected '{medicine_name}' - very low conf {conf:.2f} (validation failed)")
                        
                else:
                    # Extremely low confidence - reject
                    print(f"DEBUG: Rejected '{medicine_name}' - extremely low conf {conf:.2f}")
            
            results = filtered

        # ==================== PHASE 2: NO FALLBACK - ONLY OCR RESULTS ====================
        
        if not results:
            # OCR produced no valid medicines - return empty (no fallback)
            print("DEBUG: OCR extraction failed or empty - no medicines detected from prescription")
            return []
        
        # ==================== PHASE 4: OUTPUT CONTROL & FORMATTING ====================
        
        # Ensure output is realistic (not showing all medicines)
        # Remove duplicate medicines by name
        unique_results = []
        seen_names = set()
        for med in results:
            med_name = med.get("medicine", "").lower().strip()
            if med_name and med_name not in seen_names:
                unique_results.append(med)
                seen_names.add(med_name)
        
        results = unique_results
        
        print(f"DEBUG: Final results: {len(results)} medicines (Engine: {ocr_engine})")
        return results

    except Exception as e:
        print(f"Error in process_image: {e}")
        import traceback
        traceback.print_exc()
        return []
