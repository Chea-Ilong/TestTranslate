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
    Handle translation form submission using HuggingFace neural translation.
    """
    if request.method == 'POST':
        english_text = request.POST.get('english_text', '').strip()
        
        if not english_text:
            messages.error(request, 'Please enter some English text to translate.')
            return redirect('core:home')
        
        if len(english_text) > 5000:  # Limit text length
            messages.error(request, 'Text is too long. Please limit to 5000 characters.')
            return redirect('core:home')
        
        try:
            # Use HuggingFace translation (the only translation method)
            if HUGGINGFACE_AVAILABLE:
                logger.info(f"Starting HUGGINGFACE NEURAL translation for text: {english_text[:50]}...")
                translation_result = translate_english_to_khmer_huggingface(english_text)
                messages.success(request, f'🤗 HuggingFace Neural Translation completed! Model: {translation_result.model_used}, Confidence: {translation_result.accuracy_percentage}%, Time: {translation_result.processing_time:.3f}s')
            else:
                messages.error(request, 'HuggingFace translation service is not available.')
                return redirect('core:home')
            
            # Save to database
            translation = Translation.objects.create(
                english_text=english_text,
                khmer_text=translation_result.translated_text,
                confidence_score=translation_result.confidence_score,
                accuracy_percentage=translation_result.accuracy_percentage
            )
            
            logger.info(f"Translation saved to database with ID: {translation.id}")
            
            # Prepare context for rendering
            context = {
                'translation_successful': True,
                'english_text': english_text,
                'khmer_text': translation_result.translated_text,
                'confidence_score': translation_result.confidence_score,
                'accuracy_percentage': translation_result.accuracy_percentage,
                'translation_method': translation_result.method,
                'model_used': translation_result.model_used,
                'processing_time': translation_result.processing_time,
                'huggingface_available': HUGGINGFACE_AVAILABLE,
                
                # Confidence level for display
                'confidence_level': 'High' if translation_result.confidence_score >= 0.8 else 'Medium' if translation_result.confidence_score >= 0.6 else 'Low',
                'confidence_color': 'success' if translation_result.confidence_score >= 0.8 else 'warning' if translation_result.confidence_score >= 0.6 else 'danger',
            }
            
            # Add HuggingFace model info if available
            if HUGGINGFACE_AVAILABLE:
                try:
                    context['huggingface_model_info'] = get_huggingface_model_info()
                except Exception as e:
                    logger.warning(f"Could not get HuggingFace model info: {e}")
            
            return render(request, 'core/home.html', context)
            
        except Exception as e:
            logger.error(f"Translation failed with error: {e}")
            messages.error(request, f'Translation failed: {str(e)}')
            return redirect('core:home')
    
    return redirect('core:home')

def history(request):
    """
    Display translation history.
    """
    translations = Translation.objects.all().order_by('-created_at')
    
    context = {
        'translations': translations,
        'huggingface_available': HUGGINGFACE_AVAILABLE,
    }
    
    return render(request, 'core/history.html', context)
