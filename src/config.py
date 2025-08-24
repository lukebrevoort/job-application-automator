"""
Configuration module for the job application automator.

This module provides centralized configuration for token optimization
and other system settings.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TokenOptimizationConfig:
    """Configuration for token optimization settings."""
    enabled: bool = True
    max_resume_tokens: int = 3000
    max_job_description_tokens: int = 600
    max_cover_letter_context_tokens: int = 400
    max_fit_assessment_tokens: int = 500
    preserve_latex_structure: bool = True
    aggressive_optimization: bool = False

@dataclass
class ModelConfig:
    """Configuration for model selection and settings."""
    default_model: str = "gpt-oss:20b"
    parsing_model: str = "llama3.2:3b"
    fit_assessment_model: str = "llama3.2:3b"
    timeout: int = 120
    max_retries: int = 3
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4000

@dataclass
class SystemConfig:
    """Main system configuration."""
    token_optimization: TokenOptimizationConfig
    models: ModelConfig
    debug_mode: bool = False
    save_optimization_logs: bool = True
    
    @classmethod
    def default(cls) -> 'SystemConfig':
        """Create default configuration."""
        return cls(
            token_optimization=TokenOptimizationConfig(),
            models=ModelConfig(),
            debug_mode=False,
            save_optimization_logs=True
        )
    
    @classmethod
    def memory_optimized(cls) -> 'SystemConfig':
        """Create memory-optimized configuration for resource-constrained environments."""
        return cls(
            token_optimization=TokenOptimizationConfig(
                enabled=True,
                max_resume_tokens=2000,
                max_job_description_tokens=400,
                max_cover_letter_context_tokens=300,
                max_fit_assessment_tokens=350,
                aggressive_optimization=True
            ),
            models=ModelConfig(
                default_model="llama3.2:3b",  # Use smaller model
                parsing_model="llama3.2:3b",
                fit_assessment_model="llama3.2:3b",
                max_tokens=2000
            ),
            debug_mode=False,
            save_optimization_logs=True
        )
    
    @classmethod
    def high_quality(cls) -> 'SystemConfig':
        """Create high-quality configuration for maximum output quality."""
        return cls(
            token_optimization=TokenOptimizationConfig(
                enabled=True,
                max_resume_tokens=4000,
                max_job_description_tokens=800,
                max_cover_letter_context_tokens=600,
                max_fit_assessment_tokens=700,
                aggressive_optimization=False
            ),
            models=ModelConfig(
                default_model="gpt-oss:20b",
                parsing_model="llama3.2:3b",
                fit_assessment_model="gpt-oss:20b",  # Use larger model for assessment
                max_tokens=6000
            ),
            debug_mode=False,
            save_optimization_logs=True
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'token_optimization': {
                'enabled': self.token_optimization.enabled,
                'max_resume_tokens': self.token_optimization.max_resume_tokens,
                'max_job_description_tokens': self.token_optimization.max_job_description_tokens,
                'max_cover_letter_context_tokens': self.token_optimization.max_cover_letter_context_tokens,
                'max_fit_assessment_tokens': self.token_optimization.max_fit_assessment_tokens,
                'preserve_latex_structure': self.token_optimization.preserve_latex_structure,
                'aggressive_optimization': self.token_optimization.aggressive_optimization
            },
            'models': {
                'default_model': self.models.default_model,
                'parsing_model': self.models.parsing_model,
                'fit_assessment_model': self.models.fit_assessment_model,
                'timeout': self.models.timeout,
                'max_retries': self.models.max_retries,
                'temperature': self.models.temperature,
                'top_p': self.models.top_p,
                'max_tokens': self.models.max_tokens
            },
            'debug_mode': self.debug_mode,
            'save_optimization_logs': self.save_optimization_logs
        }
