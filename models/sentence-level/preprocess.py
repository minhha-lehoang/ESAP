import re
def isolate(text, chars):
    for c in chars:
        text = text.replace(c, f" {c} ")
    return text

def expandAbbreviation(text):
    abbr = {'pt': 'patient', 'od': 'overdose', 'o.d': 'overdose', 'yo': 'year old', 'y/o': 'year old', 
           'y.o': 'year old', 'h/o': 'history of', 'r/o': 'rule out', 's/p': 'status post', 'hx': 'history', 
            'dx': 'diagnosis', 'sx': 'symptoms', 'w/o': 'without', 'c/w': 'consistent with',
            'w/': 'with', 'u/o': 'urine output', 'd/t': 'due to','dr.': 'doctor', 'o.d.': 'overdose',
           'd/c': 'discontinued', 'c/o': 'complains of', 'neg': 'negative', 'u/s': 'ultrasound',
           'a.m': 'am', 'p.m': 'pm', 'b/c': 'because', 'o/d': 'overdose', 'sa': 'suicide attempt',
           'si': 'suicide ideation', 'r/t': 'related to', 'pmhx': 'past medical history',
           'a/w': 'associated with', 'pmh': 'past medical history', 'dtr': 'daughter', 'dgt': 'daughter',
           'd/c': 'discharge', 'nsg': 'nursing', 'fhpa': 'family history positive for',
           'r.o.s.': 'reason for admission status', 'uo': 'urine output', 'npo': 'nothing by mouth',
           'abd': 'abdomen', 'meds': 'medications', 'ods': 'overdoses', 'sucidal': 'suicidal',
           "didn't": 'did not', "don't": 'do not', "can't": 'can not', 'cannot': 'can not', 
            "doesn't": 'does not', "couldn't": 'could not'}

    text = ' ' + text + ' '
    
    for w, neww in abbr.items():
        if w in text:
            text = re.sub(r"(?<![a-zA-Z])" + w + r"(?!\w)", ' '+neww, text)
    return text

def preprocess_text(text):
    text = str(text).lower()
    text = expandAbbreviation(text).strip()
    if not text[-1] in "!,./:;?":
        text = text + "."
    text = text.replace("[**", "")
    text = text.replace("**]", "")
    text = isolate(text, "~!@#$%^&*()_+-={}:\";',./<>?\\|`'")
    text = re.sub(r" +", " ", text).strip()
    return text