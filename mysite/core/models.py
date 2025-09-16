from django.db import models
from django.utils import timezone

class Translation(models.Model):
    english_text = models.TextField(help_text="Original English text to translate")
    khmer_text = models.TextField(help_text="Translated Khmer text")
    confidence_score = models.FloatField(
        default=0.0, 
        help_text="Translation confidence score (0.0 to 1.0)"
    )
    accuracy_percentage = models.FloatField(
        default=0.0,
        help_text="Translation accuracy as percentage (0.0 to 100.0)"
    )
    timestamp = models.DateTimeField(default=timezone.now, help_text="When the translation was performed")
    
    class Meta:
        ordering = ['-timestamp']  # Most recent translations first
        
    def __str__(self):
        return f"Translation at {self.timestamp.strftime('%Y-%m-%d %H:%M')}: {self.english_text[:50]}... ({self.accuracy_percentage}%)"
    
    @property
    def confidence_level(self):
        """Return a human-readable confidence level."""
        if self.accuracy_percentage >= 85:
            return "High"
        elif self.accuracy_percentage >= 70:
            return "Good"
        elif self.accuracy_percentage >= 50:
            return "Medium"
        else:
            return "Low"
    
    @property
    def confidence_color(self):
        """Return a color class for the confidence level."""
        if self.accuracy_percentage >= 85:
            return "success"
        elif self.accuracy_percentage >= 70:
            return "primary"
        elif self.accuracy_percentage >= 50:
            return "warning"
        else:
            return "danger"
