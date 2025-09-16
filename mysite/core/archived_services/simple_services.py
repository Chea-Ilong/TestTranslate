"""
Ultra-simple translation service for testing.
"""
import logging

logger = logging.getLogger(__name__)

class TranslationResult:
    """Class to hold translation results with confidence score."""
    def __init__(self, translated_text: str, confidence_score: float, method: str = "basic"):
        self.translated_text = translated_text
        self.confidence_score = confidence_score
        self.accuracy_percentage = round(confidence_score * 100, 1)
        self.method = method

def translate_english_to_khmer(text: str) -> TranslationResult:
    """
    Basic word replacement for testing.
    """
    if not text or not text.strip():
        return TranslationResult("", 0.0, "empty")
    
    # Basic word replacements
    replacements = {
        'hi': 'សួស្តី',
        'hello': 'សួស្តី',
        'hey': 'សួស្តី',
        'world': 'ពិភពលោក',
        'good': 'ល្អ',
        'morning': 'ព្រឹក',
        'afternoon': 'រសៀល',
        'evening': 'ល្ងាច',
        'night': 'យប់',
        'thank you': 'អរគុណ',
        'thanks': 'អរគុណ',
        'please': 'សូម',
        'sorry': 'សុំទោស',
        'excuse me': 'សុំទោស',
        'yes': 'បាទ',
        'no': 'ទេ',
        'okay': 'យល់ព្រម',
        'ok': 'យល់ព្រម',
        'water': 'ទឹក',
        'food': 'អាហារ',
        'rice': 'បាយ',
        'house': 'ផ្ទះ',
        'home': 'ផ្ទះ',
        'school': 'សាលា',
        'work': 'ការងារ',
        'i am going to': 'ខ្ញុំកំពុងទៅ',
        'going to': 'កំពុងទៅ',
        'going': 'ទៅ',
        'go': 'ទៅ',
        'come': 'មក',
        'coming': 'មក',
        'eat': 'ញ៉ាំ',
        'eating': 'កំពុងញ៉ាំ',
        'drink': 'ផឹក',
        'drinking': 'កំពុងផឹក',
        'see': 'មើល',
        'look': 'មើល',
        'buy': 'ទិញ',
        'sell': 'លក់',
        'love': 'ស្រលាញ់',
        'like': 'ចូលចិត្ត',
        'friend': 'មិត្តភក្តិ',
        'family': 'គ្រួសារ',
        'mother': 'ម្តាយ',
        'father': 'ឪពុក',
        'i am': 'ខ្ញុំជា',
        'i': 'ខ្ញុំ',
        'you are': 'អ្នកជា',
        'you': 'អ្នក',
        'he is': 'គាត់ជា',
        'she is': 'នាងជា',
        'we are': 'យើងជា',
        'they are': 'ពួកគេជា',
        'to': 'ទៅ',
        'at': 'នៅ',
        'in': 'ក្នុង',
        'on': 'លើ',
        'with': 'ជាមួយ',
        'how are you': 'សុខសប្បាយទេ',
        'what is your name': 'អ្នកឈ្មោះអី',
        'my name is': 'ខ្ញុំឈ្មោះ',
        'where are you from': 'អ្នកមកពីណា',
        'i am from': 'ខ្ញុំមកពី'
    }
    
    result = text.lower().strip()
    confidence = 0.75
    words_translated = 0
    total_words = len(result.split())
    
    # Sort replacements by length (longest first) to handle phrases like "thank you" before "thank"
    sorted_replacements = sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True)
    
    for eng, khm in sorted_replacements:
        if eng in result:
            result = result.replace(eng, khm)
            confidence += 0.05
            words_translated += 1
    
    # Calculate better confidence based on word coverage
    if total_words > 0:
        coverage = words_translated / total_words
        confidence = 0.6 + (coverage * 0.35)  # Base 60% + up to 35% for coverage
    
    # Add Khmer period if not present
    if not result.endswith('។'):
        result += '។'
    
    return TranslationResult(result, min(0.95, confidence), "basic-enhanced")

# Indicate simple mode is active
ADVANCED_AVAILABLE = False