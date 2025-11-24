# src/map_polarity_to_emotion.py
"""
Deterministic mapping from TextBlob polarity -> one of the 8 ICE emotions.
Implements the rules summarized in Table 11 of the manuscript.
"""
import re

SURPRISE_REGEX = re.compile(r'\b(wow|unbelievable|incredible|no way|what a|what an|oh my|oh wow)\b', re.I)
FEAR_CUES = {"afraid","scared","fear","worry","worried","what if"}
DISGUST_CUES = {"gross","disgusting","repulsive","i can't stand","cannot stand","cant stand"}
ANGER_CUES = {"hate","stupid","idiot","damn","destroy","kill","screw you","idiots","urgh"}
SAD_CUES = {"sad","regret","sorry","lost","depressed","tear","sorrow"}
MIXED_CUES = {"but","however","although","though"}  # simple indicator for mixed context
ASTONISH_CUES = {"wow","unbelievable","amazing","incredible","no way","what a","what an"}

def contains_any(text, cue_set):
    txt = (text or "").lower()
    for c in cue_set:
        if c in txt:
            return True
    return False

def map_polarity_to_emotion(polarity, text):
    """
    Map polarity (float in [-1,1]) and text (string) to one of:
    ["Happy","Sad","Angry","Fearful","Disgust","Surprised","Neutral","Mixed"]
    """
    try:
        p = float(polarity)
    except:
        p = 0.0
    txt = text or ""
    txt_low = txt.lower()

    # Strong positive
    if p >= 0.6:
        if SURPRISE_REGEX.search(txt) or contains_any(txt, ASTONISH_CUES):
            return "Surprised"
        return "Happy"

    # Moderate positive
    if 0.2 <= p < 0.6:
        if contains_any(txt, MIXED_CUES):
            return "Mixed"
        return "Happy"

    # Neutral
    if -0.2 < p < 0.2:
        return "Neutral"

    # Moderate negative
    if -0.6 < p <= -0.2:
        if contains_any(txt, FEAR_CUES) or "?" in txt:
            return "Fearful"
        return "Sad"

    # Strong negative
    if p <= -0.6:
        if contains_any(txt, DISGUST_CUES):
            return "Disgust"
        if contains_any(txt, ANGER_CUES):
            return "Angry"
        return "Angry"

    return "Neutral"
