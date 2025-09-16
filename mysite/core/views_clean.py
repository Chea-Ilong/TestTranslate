from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect
from .models import Translation

import logging

logger = logging.getLogger(__name__)

# Import HuggingFace translation service (the only one we use)
try:
    from .safe_huggingface_services import translate_english_to_khmer_huggingface, get_huggingface_model_info
    HUGGINGFACE_AVAILABLE = True
    logger.info("🤗 HuggingFace neural translation service loaded")
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.error("HuggingFace translation service not available")

def home(request):
    """
    Home view that displays the translation form and recent translations.
    """
    # Get recent translations for display (last 10)
    recent_translations = Translation.objects.all()[:10]
    
    context = {
        'recent_translations': recent_translations,
        'huggingface_available': HUGGINGFACE_AVAILABLE,
    }
    
    # Add HuggingFace model info if available
    if HUGGINGFACE_AVAILABLE:
        try:
            context['huggingface_model_info'] = get_huggingface_model_info()
        except Exception as e:
            logger.warning(f"Could not get HuggingFace model info: {e}")
    
    return render(request, 'core/home.html', context)

@csrf_protect
def translate_text(request):
    """
    Handle translation form submission and display results.
    """
    if request.method == 'POST':
        english_text = request.POST.get('english_text', '').strip()
        use_huggingface = request.POST.get('huggingface_mode', False) == 'on'
        use_safe_optimized = request.POST.get('safe_optimized_mode', False) == 'on'
        use_ai_mode = request.POST.get('ai_mode', False) == 'on'
        
        if not english_text:
            messages.error(request, 'Please enter some English text to translate.')
            return redirect('core:home')
        
        if len(english_text) > 5000:  # Limit text length
            messages.error(request, 'Text is too long. Please limit to 5000 characters.')
            return redirect('core:home')
        
        try:
            # Choose translation method based on user preference and availability
            # Priority: HuggingFace > Safe Optimized > AI Enhanced > Simple
            
            if use_huggingface and HUGGINGFACE_AVAILABLE and HUGGINGFACE_SERVICE_AVAILABLE:
                logger.info(f"Starting HUGGINGFACE NEURAL translation for text: {english_text[:50]}...")
                translation_result = translate_english_to_khmer_huggingface(english_text)
                messages.success(request, f'🤗 HuggingFace Neural Translation completed! Model: {translation_result.model_used}, Confidence: {translation_result.accuracy_percentage}%, Time: {translation_result.processing_time:.3f}s')
                
            elif use_safe_optimized and SAFE_OPTIMIZED_AVAILABLE:
                logger.info(f"Starting SAFE OPTIMIZED translation for text: {english_text[:50]}...")
                translation_dict = translate_english_to_khmer_safe_optimized(english_text)
                
                # Create result object from dictionary
                class TranslationResult:
                    def __init__(self, trans_dict):
                        self.translated_text = trans_dict.get('translated_text', '')
                        self.confidence_score = trans_dict.get('confidence', 0.0)
                        self.accuracy_percentage = round(self.confidence_score * 100, 1)
                        self.method = f"safe-optimized-{trans_dict.get('service_used', 'unknown')}"
                        self.cached = trans_dict.get('cached', False)
                        self.processing_time = trans_dict.get('processing_time', 0.0)
                
                translation_result = TranslationResult(translation_dict)
                cache_info = " (cached)" if translation_result.cached else ""
                messages.success(request, f'🚀 Safe Optimized Translation completed{cache_info}! Service: {translation_dict.get("service_used")}, Confidence: {translation_result.accuracy_percentage}%')
                
            elif use_ai_mode and AI_ENHANCED_AVAILABLE:
                logger.info(f"Starting AI ENHANCED translation for text: {english_text[:50]}...")
                translation_result = translate_english_to_khmer_ai_enhanced(english_text)
                messages.success(request, f'🤖 AI Enhanced Translation completed! Method: {translation_result.method}, Confidence: {translation_result.accuracy_percentage}%')
                
            elif HUGGINGFACE_AVAILABLE and HUGGINGFACE_SERVICE_AVAILABLE:
                # Auto-fallback to HuggingFace if no specific mode selected but available
                logger.info(f"Auto-selecting HUGGINGFACE NEURAL translation for text: {english_text[:50]}...")
                translation_result = translate_english_to_khmer_huggingface(english_text)
                messages.success(request, f'🤗 HuggingFace Neural Translation (auto-selected) completed! Model: {translation_result.model_used}, Confidence: {translation_result.accuracy_percentage}%')
                
            elif SIMPLE_AVAILABLE:
                logger.info(f"Starting SIMPLE translation for text: {english_text[:50]}...")
                translation_result = translate_english_to_khmer(english_text)
                messages.success(request, f'✅ Simple translation completed! Confidence: {translation_result.accuracy_percentage}%')
                
            else:
                messages.error(request, 'No translation services are available.')
                return redirect('core:home')
            
            # Save to database
            translation = Translation.objects.create(
                english_text=english_text,
                khmer_text=translation_result.translated_text,
                confidence_score=translation_result.confidence_score,
                accuracy_percentage=translation_result.accuracy_percentage
            )
            
            # Get recent translations for display
            recent_translations = Translation.objects.all()[:10]
            
            context = {
                'english_text': english_text,
                'khmer_text': translation_result.translated_text,
                'confidence_score': translation_result.confidence_score,
                'accuracy_percentage': translation_result.accuracy_percentage,
                'confidence_level': translation.confidence_level,
                'confidence_color': translation.confidence_color,
                'translation_method': translation_result.method,
                'translation_successful': True,
                'recent_translations': recent_translations,
                'safe_optimized_available': SAFE_OPTIMIZED_AVAILABLE,
                'ai_enhanced_available': AI_ENHANCED_AVAILABLE,
                'simple_available': SIMPLE_AVAILABLE,
                'used_safe_optimized': use_safe_optimized,
                'used_ai_mode': use_ai_mode,
            }
            
            # Add processing details for safe optimized
            if use_safe_optimized and hasattr(translation_result, 'processing_time'):
                context['processing_time'] = translation_result.processing_time
                context['cached'] = getattr(translation_result, 'cached', False)
            
            # Add system status if available
            if SAFE_OPTIMIZED_AVAILABLE:
                try:
                    context['system_status'] = get_translation_system_status()
                except Exception as e:
                    logger.warning(f"Could not get system status: {e}")
            
            return render(request, 'core/home.html', context)
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            messages.error(request, f'Translation failed: {str(e)}')
            return redirect('core:home')
    
    # If not POST, redirect to home
    return redirect('core:home')

def translation_history(request):
    """
    Display translation history page.
    """
    translations = Translation.objects.all()[:50]  # Last 50 translations
    
    context = {
        'translations': translations,
    }
    
    return render(request, 'core/history.html', context)