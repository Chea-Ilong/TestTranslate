#!/usr/bin/env python3
"""
Safe Optimized Translation Services
Provides caching, batching, and memory monitoring without PyTorch dependencies
"""

import sys
import os
import logging
import time
import threading
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("psutil not available, memory monitoring disabled")
    PSUTIL_AVAILABLE = False

@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata"""
    value: Any
    timestamp: float
    access_count: int = 0
    
    @property
    def age(self) -> float:
        return time.time() - self.timestamp
    
    def is_expired(self, ttl: float) -> bool:
        return self.age > ttl

class TranslationCache:
    """Advanced LRU cache with TTL and memory management"""
    
    def __init__(self, max_size: int = 2000, ttl: float = 7200.0):  # 2 hours TTL
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired': 0
        }
        self._lock = threading.RLock()
    
    def _evict_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if entry.is_expired(self.ttl):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            self.stats['expired'] += 1
    
    def _evict_lru(self):
        """Evict least recently used entries"""
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest
            self.stats['evictions'] += 1
    
    def get(self, key: str) -> Optional[str]:
        """Get cached translation"""
        with self._lock:
            self._evict_expired()
            
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired(self.ttl):
                    # Move to end (most recently used)
                    entry.access_count += 1
                    self.cache.move_to_end(key)
                    self.stats['hits'] += 1
                    return entry.value
                else:
                    del self.cache[key]
                    self.stats['expired'] += 1
            
            self.stats['misses'] += 1
            return None
    
    def put(self, key: str, value: str):
        """Cache translation result"""
        with self._lock:
            self._evict_expired()
            self._evict_lru()
            
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                access_count=1
            )
            self.cache[key] = entry
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests) if total_requests > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'expired': self.stats['expired']
            }
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self.cache.clear()
            self.stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'expired': 0
            }

class MemoryMonitor:
    """System memory monitoring"""
    
    def __init__(self):
        self.available = PSUTIL_AVAILABLE
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage"""
        if not self.available:
            return {
                'total_gb': 0.0,
                'available_gb': 0.0,
                'used_gb': 0.0,
                'percent': 0.0,
                'available': False
            }
        
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent,
                'available': True
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {
                'total_gb': 0.0,
                'available_gb': 0.0,
                'used_gb': 0.0,
                'percent': 0.0,
                'available': False
            }
    
    def has_sufficient_memory(self, required_gb: float = 1.0) -> bool:
        """Check if sufficient memory is available"""
        if not self.available:
            return True  # Assume OK if we can't check
        
        info = self.get_memory_info()
        return info.get('available_gb', 0) >= required_gb

class SafeOptimizedTranslationService:
    """
    Safe optimized translation service with caching and monitoring
    Uses AI-enhanced fallback without PyTorch dependencies
    """
    
    def __init__(self):
        self.cache = TranslationCache(max_size=2000, ttl=7200.0)
        self.memory_monitor = MemoryMonitor()
        
        # Load AI-enhanced fallback safely
        try:
            from .ai_enhanced_services import translate_english_to_khmer_ai_enhanced
            self.ai_fallback = translate_english_to_khmer_ai_enhanced
            logger.info("AI-enhanced translation service loaded successfully")
        except ImportError:
            logger.warning("Could not import AI-enhanced service, using simple fallback")
            self.ai_fallback = None
        
        # Load simple fallback
        try:
            from .simple_services import translate_english_to_khmer
            self.simple_fallback = translate_english_to_khmer
            logger.info("Simple translation service loaded as fallback")
        except ImportError:
            logger.error("Could not import simple translation service")
            self.simple_fallback = None
        
        self.batch_size = 32  # Process in batches
        logger.info("Safe optimized translation service initialized")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return f"en_km_{hash(text.strip().lower())}"
    
    def _translate_single(self, text: str) -> Dict[str, Any]:
        """Translate single text with fallback chain"""
        if not text or not text.strip():
            return {
                'translated_text': '',
                'confidence': 0.0,
                'service_used': 'none',
                'cached': False
            }
        
        text = text.strip()
        cache_key = self._get_cache_key(text)
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return {
                'translated_text': cached_result,
                'confidence': 1.0,  # Assume cached results are good
                'service_used': 'cache',
                'cached': True
            }
        
        # Try AI-enhanced service first
        if self.ai_fallback:
            try:
                result = self.ai_fallback(text)
                if result and hasattr(result, 'translated_text'):
                    translation = result.translated_text
                    confidence = result.confidence_score
                    
                    # Cache the result
                    self.cache.put(cache_key, translation)
                    
                    return {
                        'translated_text': translation,
                        'confidence': confidence,
                        'service_used': 'ai_enhanced',
                        'cached': False
                    }
            except Exception as e:
                logger.warning(f"AI-enhanced service failed: {e}")
        
        # Fall back to simple service
        if self.simple_fallback:
            try:
                result = self.simple_fallback(text)
                if result and hasattr(result, 'translated_text'):
                    translation = result.translated_text
                    confidence = result.confidence_score
                    
                    # Cache the result
                    self.cache.put(cache_key, translation)
                    
                    return {
                        'translated_text': translation,
                        'confidence': confidence,
                        'service_used': 'simple',
                        'cached': False
                    }
            except Exception as e:
                logger.error(f"Simple service failed: {e}")
        
        # Ultimate fallback
        return {
            'translated_text': text,  # Return original text
            'confidence': 0.0,
            'service_used': 'none',
            'cached': False
        }
    
    def translate(self, input_text: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Translate text with caching and optimization
        
        Args:
            input_text: Single string or list of strings to translate
            
        Returns:
            Single translation result or list of results
        """
        start_time = time.time()
        
        # Handle single string
        if isinstance(input_text, str):
            result = self._translate_single(input_text)
            result['processing_time'] = time.time() - start_time
            result['memory_info'] = self.memory_monitor.get_memory_info()
            return result
        
        # Handle list (batch processing)
        if isinstance(input_text, list):
            results = []
            
            # Process in batches for memory efficiency
            for i in range(0, len(input_text), self.batch_size):
                batch = input_text[i:i + self.batch_size]
                batch_results = []
                
                for text in batch:
                    result = self._translate_single(text)
                    batch_results.append(result)
                
                results.extend(batch_results)
                
                # Check memory after each batch
                if not self.memory_monitor.has_sufficient_memory(0.5):
                    logger.warning("Low memory detected, may need to clear cache")
            
            # Add batch metadata
            total_time = time.time() - start_time
            for i, result in enumerate(results):
                result['batch_index'] = i
                result['batch_size'] = len(input_text)
                result['total_processing_time'] = total_time
            
            return results
        
        # Invalid input
        raise ValueError("Input must be string or list of strings")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'cache_stats': self.cache.get_stats(),
            'memory_info': self.memory_monitor.get_memory_info(),
            'services_available': {
                'ai_enhanced': self.ai_fallback is not None,
                'simple': self.simple_fallback is not None,
            },
            'batch_size': self.batch_size,
            'timestamp': time.time()
        }
    
    def clear_cache(self):
        """Clear translation cache"""
        self.cache.clear()
        logger.info("Translation cache cleared")

# Global service instance
_service_instance = None
_service_lock = threading.Lock()

def get_safe_optimized_service() -> SafeOptimizedTranslationService:
    """Get singleton service instance"""
    global _service_instance
    
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = SafeOptimizedTranslationService()
    
    return _service_instance

def translate_english_to_khmer_safe_optimized(text: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Safe optimized translation function with caching and monitoring
    
    Args:
        text: English text to translate (string or list)
        
    Returns:
        Translation result with metadata
    """
    service = get_safe_optimized_service()
    return service.translate(text)

def get_translation_system_status() -> Dict[str, Any]:
    """Get translation system status"""
    service = get_safe_optimized_service()
    return service.get_system_status()

def clear_translation_cache():
    """Clear translation cache"""
    service = get_safe_optimized_service()
    service.clear_cache()

if __name__ == "__main__":
    # Test the service
    print("Testing Safe Optimized Translation Service")
    print("=" * 50)
    
    # Test single translation
    result = translate_english_to_khmer_safe_optimized("Hello, how are you?")
    print(f"Single translation: {result}")
    
    # Test caching
    result2 = translate_english_to_khmer_safe_optimized("Hello, how are you?")
    print(f"Cached translation: {result2}")
    
    # Test batch
    batch_result = translate_english_to_khmer_safe_optimized([
        "Hello", "Thank you", "Goodbye", "Please", "Yes"
    ])
    print(f"Batch translation: {len(batch_result)} results")
    
    # Get system status
    status = get_translation_system_status()
    print(f"System status: {status}")