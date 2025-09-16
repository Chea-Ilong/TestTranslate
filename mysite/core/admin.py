from django.contrib import admin
from .models import Translation

@admin.register(Translation)
class TranslationAdmin(admin.ModelAdmin):
    list_display = ('english_text_preview', 'khmer_text_preview', 'accuracy_percentage', 'confidence_level', 'timestamp')
    list_filter = ('timestamp', 'confidence_score')
    search_fields = ('english_text', 'khmer_text')
    readonly_fields = ('timestamp', 'confidence_level')
    ordering = ('-timestamp',)
    
    def english_text_preview(self, obj):
        return obj.english_text[:50] + "..." if len(obj.english_text) > 50 else obj.english_text
    english_text_preview.short_description = "English Text"
    
    def khmer_text_preview(self, obj):
        return obj.khmer_text[:50] + "..." if len(obj.khmer_text) > 50 else obj.khmer_text
    khmer_text_preview.short_description = "Khmer Text"
