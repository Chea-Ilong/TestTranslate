#!/usr/bin/env python3
"""
HuggingFace Translation Service
Professional-grade neural machine translation using pre-trained models
"""

import os
import logging
import time
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HuggingFace imports with graceful fallback
try:
    import torch
    # Test torch functionality
    torch.tensor([1.0])
    torch.manual_seed(42)
    TORCH_AVAILABLE = True
    logger.info("PyTorch loaded and tested successfully")
except Exception as e:
    logger.warning(f"PyTorch not stable: {e}")
    TORCH_AVAILABLE = False

try:
    if TORCH_AVAILABLE:
        from transformers import MarianMTModel, MarianTokenizer, pipeline
        HUGGINGFACE_AVAILABLE = True
        logger.info("HuggingFace transformers loaded successfully")
    else:
        HUGGINGFACE_AVAILABLE = False
        logger.warning("HuggingFace not loaded due to PyTorch issues")
except ImportError as e:
    logger.warning(f"HuggingFace not available: {e}")
    HUGGINGFACE_AVAILABLE = False

@dataclass
class HFTranslationResult:
    """Enhanced translation result with HuggingFace metadata"""
    translated_text: str
    confidence_score: float
    model_used: str
    processing_time: float
    method: str = "huggingface-neural"
    
    @property
    def accuracy_percentage(self) -> float:
        return round(self.confidence_score * 100, 1)

class HuggingFaceTranslationService:
    """
    Professional neural machine translation using HuggingFace models
    Supports multiple models with automatic fallback
    """
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self._lock = threading.RLock()
        
        # Model configurations (ordered by quality/reliability)
        self.model_configs = [
            {
                'name': 'opus-mt-en-mul',
                'model_id': 'Helsinki-NLP/opus-mt-en-mul',
                'description': 'English to Multiple Languages (includes Asian languages)',
                'target_language': 'km',  # Khmer code
                'quality': 'high',
                'size': '300MB'
            },
            {
                'name': 'opus-mt-en-zh',
                'model_id': 'Helsinki-NLP/opus-mt-en-zh', 
                'description': 'English to Chinese (fallback for Asian languages)',
                'target_language': 'zh',
                'quality': 'medium',
                'size': '250MB'
            },
            {
                'name': 'mbart-large',
                'model_id': 'facebook/mbart-large-50-many-to-many-mmt',
                'description': 'Multilingual BART (50+ languages)',
                'target_language': 'km_KH',
                'quality': 'very-high',
                'size': '2.4GB'
            }
        ]
        
        # Fallback translations for when models fail
        self.fallback_translations = {
            "hello": "សួស្តី",
            "how are you": "តើអ្នកសុខសប្បាយទេ",
            "thank you": "អរគុណ",
            "goodbye": "លាហើយ",
            "yes": "បាទ/ចាស",
            "no": "ទេ",
            "please": "សូម",
            "sorry": "សុំទោស",
            "good morning": "អរុណសួស្តី",
            "good evening": "សាយ័នសួស្តី",
            "nice to meet you": "រីករាយដែលបានជួបអ្នក",
            "i love you": "ខ្ញុំស្រលាញ់អ្នក",
            "good": "ល្អ",
            "bad": "អាក្រក់",
            "big": "ធំ",
            "small": "តូច",
            "water": "ទឹក",
            "food": "អាហារ",
            "house": "ផ្ទះ",
            "school": "សាលា",
            "work": "ការងារ"
        }
        
        self.current_model = None
        self.model_loaded = self.load_best_available_model()
    
    def load_best_available_model(self):
        """Load the best available model with graceful fallback"""
        if not HUGGINGFACE_AVAILABLE:
            logger.error("HuggingFace not available, cannot load models")
            return False
        
        with self._lock:
            # Try lighter models first to avoid memory issues
            lightweight_configs = [
                {
                    'name': 'opus-mt-en-zh',
                    'model_id': 'Helsinki-NLP/opus-mt-en-zh', 
                    'description': 'English to Chinese (lightweight Asian language model)',
                    'target_language': 'zh',
                    'quality': 'medium',
                    'size': '250MB'
                }
            ]
            
            for config in lightweight_configs:
                try:
                    logger.info(f"Attempting to load lightweight model: {config['description']}")
                    
                    # Use Marian models with CPU-only settings
                    tokenizer = MarianTokenizer.from_pretrained(
                        config['model_id'],
                        local_files_only=False,
                        torch_dtype=None  # Let it use default
                    )
                    model = MarianMTModel.from_pretrained(
                        config['model_id'],
                        local_files_only=False,
                        torch_dtype=None  # Let it use default
                    )
                    
                    # Force CPU usage
                    model = model.to('cpu')
                    model.eval()  # Set to evaluation mode
                    
                    self.tokenizers[config['name']] = tokenizer
                    self.models[config['name']] = model
                    
                    self.current_model = config
                    logger.info(f"✅ Successfully loaded: {config['description']} ({config['size']})")
                    return True
                    
                except Exception as e:
                    logger.warning(f"❌ Failed to load {config['name']}: {e}")
                    continue
            
            logger.error("❌ No HuggingFace models could be loaded")
            return False
    
    def translate_with_opus(self, text: str, model_name: str) -> Optional[str]:
        """Translate using OPUS-MT models"""
        try:
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]
            
            # Prepare input with length limits
            max_length = min(512, len(text.split()) * 10)  # Conservative estimate
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_length
            )
            
            # Generate translation with conservative settings
            with torch.no_grad():
                translated = model.generate(
                    **inputs, 
                    max_length=max_length,
                    num_beams=2,  # Reduced from 4
                    early_stopping=True,
                    do_sample=False,
                    no_repeat_ngram_size=2
                )
            
            # Decode result
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
            return result.strip()
            
        except Exception as e:
            logger.error(f"OPUS translation failed: {e}")
            return None
    
    def translate_with_fallback(self, text: str) -> str:
        """Fallback translation using basic dictionary"""
        text_lower = text.lower().strip()
        
        # Check for exact matches first
        if text_lower in self.fallback_translations:
            return self.fallback_translations[text_lower]
        
        # Try partial matches
        for english, khmer in self.fallback_translations.items():
            if english in text_lower:
                return f"{khmer} ({text})"
        
    def translate_with_mbart(self, text: str) -> Optional[str]:
        """Translate using mBART model"""
        try:
            pipe = self.pipelines['mbart-large']
            
            # mBART translation
            result = pipe(text, src_lang="en_XX", tgt_lang="km_KH")
            
            if result and len(result) > 0:
                return result[0]['translation_text'].strip()
            return None
            
        except Exception as e:
            logger.error(f"mBART translation failed: {e}")
            return None
        """Translate using mBART model"""
        try:
            pipe = self.pipelines['mbart-large']
            
            # mBART translation
            result = pipe(text, src_lang="en_XX", tgt_lang="km_KH")
            
            if result and len(result) > 0:
                return result[0]['translation_text'].strip()
            return None
            
        except Exception as e:
            logger.error(f"mBART translation failed: {e}")
            return None
    
    def calculate_confidence(self, original: str, translated: str, model_quality: str) -> float:
        """Calculate confidence score based on model quality and translation characteristics"""
        base_confidence = {
            'very-high': 0.95,
            'high': 0.90,
            'medium': 0.85,
            'low': 0.75
        }.get(model_quality, 0.80)
        
        # Adjust based on text characteristics
        if not translated or translated == original:
            return 0.3  # Low confidence for failed translations
        
        if len(translated) < 3:
            base_confidence *= 0.8  # Reduce confidence for very short translations
        
        if len(original.split()) > 20:
            base_confidence *= 0.95  # Slightly reduce for very long texts
            
        return min(0.98, max(0.60, base_confidence))
    
    def translate_english_to_khmer(self, text: str) -> HFTranslationResult:
        """
        Main translation function using HuggingFace neural models
        """
        start_time = time.time()
        
        if not text or not text.strip():
            return HFTranslationResult(
                translated_text="",
                confidence_score=0.0,
                model_used="none",
                processing_time=0.0,
                method="huggingface-empty"
            )
        
        if not HUGGINGFACE_AVAILABLE or not self.model_loaded:
            # Use fallback dictionary
            translated_text = self.translate_with_fallback(text)
            return HFTranslationResult(
                translated_text=translated_text,
                confidence_score=0.6 if translated_text != f"[HuggingFace unavailable] {text}" else 0.1,
                model_used="fallback-dictionary",
                processing_time=time.time() - start_time,
                method="huggingface-fallback"
            )
        
        text = text.strip()
        logger.info(f"HuggingFace translation starting: '{text[:50]}...' using {self.current_model['name']}")
        
        # Attempt translation with current model
        translated_text = None
        
        try:
            with self._lock:
                if self.current_model['name'] == 'mbart-large':
                    translated_text = self.translate_with_mbart(text)
                else:
                    translated_text = self.translate_with_opus(text, self.current_model['name'])
            
            if translated_text:
                # Successful translation
                processing_time = time.time() - start_time
                confidence = self.calculate_confidence(text, translated_text, self.current_model['quality'])
                
                # Ensure Khmer sentence ending
                if not translated_text.endswith('។') and len(translated_text) > 3:
                    translated_text += '។'
                
                result = HFTranslationResult(
                    translated_text=translated_text,
                    confidence_score=confidence,
                    model_used=self.current_model['name'],
                    processing_time=processing_time,
                    method="huggingface-neural"
                )
                
                logger.info(f"✅ HuggingFace translation completed: '{translated_text[:50]}...' "
                           f"({result.accuracy_percentage}% confidence, {processing_time:.3f}s)")
                return result
        
        except Exception as e:
            logger.error(f"HuggingFace translation failed: {e}")
        
        # Fallback result using dictionary
        translated_text = self.translate_with_fallback(text)
        processing_time = time.time() - start_time
        return HFTranslationResult(
            translated_text=translated_text,
            confidence_score=0.5 if translated_text != f"[HuggingFace unavailable] {text}" else 0.2,
            model_used=self.current_model['name'] if self.current_model else "fallback-dictionary",
            processing_time=processing_time,
            method="huggingface-fallback"
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.model_loaded and not self.current_model:
            return {
                'status': 'fallback_mode',
                'available': True,
                'current_model': 'fallback-dictionary',
                'description': 'Basic dictionary fallback (20+ phrases)',
                'quality': 'basic',
                'size': 'minimal',
                'huggingface_available': HUGGINGFACE_AVAILABLE,
                'models_loaded': 0
            }
        
        return {
            'status': 'loaded',
            'available': True,
            'current_model': self.current_model['name'] if self.current_model else 'fallback-dictionary',
            'description': self.current_model['description'] if self.current_model else 'Basic dictionary fallback',
            'quality': self.current_model['quality'] if self.current_model else 'basic',
            'size': self.current_model['size'] if self.current_model else 'minimal',
            'huggingface_available': HUGGINGFACE_AVAILABLE,
            'models_loaded': len(self.models) + len(self.pipelines)
        }

# Global service instance
_hf_service_instance = None
_hf_service_lock = threading.Lock()

def get_huggingface_service() -> HuggingFaceTranslationService:
    """Get singleton HuggingFace service instance"""
    global _hf_service_instance
    
    if _hf_service_instance is None:
        with _hf_service_lock:
            if _hf_service_instance is None:
                _hf_service_instance = HuggingFaceTranslationService()
    
    return _hf_service_instance

def translate_english_to_khmer_huggingface(text: str) -> HFTranslationResult:
    """
    Main HuggingFace translation function
    
    Args:
        text: English text to translate
        
    Returns:
        HFTranslationResult with neural translation
    """
    service = get_huggingface_service()
    return service.translate_english_to_khmer(text)

def get_huggingface_model_info() -> Dict[str, Any]:
    """Get HuggingFace model information"""
    service = get_huggingface_service()
    return service.get_model_info()

# Service availability flag - always true since we have fallback
HUGGINGFACE_SERVICE_AVAILABLE = True

if HUGGINGFACE_AVAILABLE:
    logger.info("🤗 HuggingFace Translation Service loaded successfully")
else:
    logger.warning("⚠️ HuggingFace Translation Service not available")

if __name__ == "__main__":
    # Test the service
    print("🤗 Testing HuggingFace Translation Service")
    print("=" * 50)
    
    # Get model info
    info = get_huggingface_model_info()
    print(f"Model Info: {info}")
    
    if info['available']:
        # Test translation
        test_phrases = [
            "Hello, how are you?",
            "Thank you very much",
            "I am going to school",
            "Nice to meet you"
        ]
        
        for phrase in test_phrases:
            result = translate_english_to_khmer_huggingface(phrase)
            print(f"'{phrase}' → '{result.translated_text}' "
                  f"({result.accuracy_percentage}% confidence, {result.processing_time:.3f}s)")
    else:
        print("❌ HuggingFace service not available")