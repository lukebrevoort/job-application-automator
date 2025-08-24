"""
Token Optimization Module

This module provides intelligent token optimization strategies to reduce context length
while maintaining high-quality output for the job application automation system.

Key optimization strategies:
1. Resume content condensation with semantic preservation
2. Job posting information extraction and summarization
3. Dynamic content prioritization based on job requirements
4. LaTeX structure preservation during optimization
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from job_scraper import JobPosting

logger = logging.getLogger(__name__)

@dataclass
class OptimizedContent:
    """Container for optimized content with metadata."""
    optimized_text: str
    original_length: int
    optimized_length: int
    compression_ratio: float
    preserved_keywords: List[str]
    optimization_strategy: str

@dataclass
class ResumeSection:
    """Represents a section of the resume with priority scoring."""
    name: str
    content: str
    start_pos: int
    end_pos: int
    priority_score: float
    keywords: Set[str]

class TokenOptimizer:
    """
    Intelligent token optimization for job application automation.
    
    This class provides methods to reduce token usage while preserving
    the essential information needed for high-quality personalization.
    """
    
    def __init__(self):
        """Initialize the token optimizer."""
        # LaTeX commands that should never be removed
        self.latex_protected_commands = {
            r'\documentclass', r'\usepackage', r'\begin{document}', r'\end{document}',
            r'\section', r'\subsection', r'\resumeSubheading', r'\resumeItem',
            r'\resumeSubHeadingListStart', r'\resumeSubHeadingListEnd',
            r'\resumeItemListStart', r'\resumeItemListEnd', r'\resumeProjectHeading',
            r'\textbf', r'\textit', r'\underline', r'\href', r'\small', r'\scshape'
        }
        
        # Keywords that should be preserved at high priority
        self.high_priority_keywords = {
            'python', 'javascript', 'react', 'node.js', 'sql', 'machine learning',
            'ai', 'llm', 'api', 'backend', 'frontend', 'full-stack', 'git',
            'aws', 'cloud', 'docker', 'kubernetes', 'mongodb', 'postgresql'
        }
    
    def optimize_resume_for_job(self, 
                               latex_resume: str, 
                               job_posting: JobPosting,
                               max_tokens: int = 3000) -> OptimizedContent:
        """
        Optimize a LaTeX resume by prioritizing content relevant to the job posting.
        
        Args:
            latex_resume: The full LaTeX resume content
            job_posting: The job posting to optimize for
            max_tokens: Maximum tokens to target (roughly 4 chars per token)
            
        Returns:
            OptimizedContent with the optimized resume
        """
        original_length = len(latex_resume)
        target_length = max_tokens * 4  # Rough chars per token estimate
        
        if original_length <= target_length:
            return OptimizedContent(
                optimized_text=latex_resume,
                original_length=original_length,
                optimized_length=original_length,
                compression_ratio=1.0,
                preserved_keywords=list(self._extract_keywords(latex_resume)),
                optimization_strategy="no_optimization_needed"
            )
        
        # Extract job keywords for prioritization
        job_keywords = self._extract_job_keywords(job_posting)
        logger.info(f"Job keywords extracted: {job_keywords}")
        
        # Parse resume sections
        sections = self._parse_resume_sections(latex_resume)
        
        # Score sections based on job relevance
        scored_sections = self._score_sections_for_job(sections, job_keywords)
        
        # Optimize content while preserving LaTeX structure
        optimized_resume = self._rebuild_optimized_resume(
            latex_resume, scored_sections, target_length, job_keywords
        )
        
        return OptimizedContent(
            optimized_text=optimized_resume,
            original_length=original_length,
            optimized_length=len(optimized_resume),
            compression_ratio=len(optimized_resume) / original_length,
            preserved_keywords=job_keywords,
            optimization_strategy="section_prioritization"
        )
    
    def condense_job_posting(self, 
                           job_posting: JobPosting,
                           max_tokens: int = 800) -> OptimizedContent:
        """
        Condense job posting information to essential details.
        
        Args:
            job_posting: The job posting to condense
            max_tokens: Maximum tokens to target
            
        Returns:
            OptimizedContent with condensed job information
        """
        # Extract key information
        essential_info = {
            "title": job_posting.title,
            "company": job_posting.company,
            "location": job_posting.location,
            "key_skills": job_posting.skills[:8],  # Top 8 skills only
            "core_requirements": self._extract_core_requirements(job_posting.requirements),
            "summary": self._summarize_description(job_posting.description, max_tokens // 2)
        }
        
        # Format condensed information
        condensed = self._format_condensed_job_info(essential_info)
        
        original_length = len(job_posting.description) + len(str(job_posting.skills)) + len(str(job_posting.requirements))
        
        return OptimizedContent(
            optimized_text=condensed,
            original_length=original_length,
            optimized_length=len(condensed),
            compression_ratio=len(condensed) / max(original_length, 1),
            preserved_keywords=essential_info["key_skills"],
            optimization_strategy="job_condensation"
        )
    
    def _extract_job_keywords(self, job_posting: JobPosting) -> Set[str]:
        """Extract relevant keywords from job posting."""
        keywords = set()
        
        # Add explicit skills
        keywords.update(skill.lower().strip() for skill in job_posting.skills)
        
        # Extract from description and requirements
        text_to_analyze = f"{job_posting.description} {' '.join(job_posting.requirements)}"
        
        # Tech stack patterns
        tech_patterns = [
            r'\b(python|javascript|react|node\.?js|sql|mongodb|postgresql|aws|docker|kubernetes)\b',
            r'\b(machine learning|ai|llm|api|backend|frontend|full-?stack)\b',
            r'\b(git|github|agile|scrum|ci/cd|devops)\b',
            r'\b(html|css|typescript|java|c\+\+|go|rust)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text_to_analyze.lower())
            keywords.update(matches)
        
        # Add high priority keywords that appear in text
        for keyword in self.high_priority_keywords:
            if keyword.lower() in text_to_analyze.lower():
                keywords.add(keyword.lower())
        
        return keywords
    
    def _parse_resume_sections(self, latex_resume: str) -> List[ResumeSection]:
        """Parse LaTeX resume into sections with metadata."""
        sections = []
        
        # Find all section boundaries using actual LaTeX \section{} commands
        section_pattern = r'\\section\{([^}]+)\}'
        section_matches = list(re.finditer(section_pattern, latex_resume))
        
        if not section_matches:
            # Fallback: return the whole resume as one section
            return [ResumeSection(
                name='complete_resume',
                content=latex_resume,
                start_pos=0,
                end_pos=len(latex_resume),
                priority_score=1.0,
                keywords=self._extract_keywords(latex_resume)
            )]
        
        for i, match in enumerate(section_matches):
            section_title = match.group(1).lower()
            start_pos = match.start()
            
            # Find the end position (start of next section or end of document)
            if i + 1 < len(section_matches):
                end_pos = section_matches[i + 1].start()
            else:
                # Last section - find end of content before \end{document}
                end_match = re.search(r'\\end{document}', latex_resume[start_pos:])
                if end_match:
                    end_pos = start_pos + end_match.start()
                else:
                    end_pos = len(latex_resume)
            
            content = latex_resume[start_pos:end_pos]
            keywords = self._extract_keywords(content)
            
            # Map section titles to standard names
            section_name = self._normalize_section_name(section_title)
            
            sections.append(ResumeSection(
                name=section_name,
                content=content,
                start_pos=start_pos,
                end_pos=end_pos,
                priority_score=0.0,  # Will be calculated later
                keywords=keywords
            ))
        
        return sections
    
    def _normalize_section_name(self, section_title: str) -> str:
        """Normalize section titles to standard names."""
        title_lower = section_title.lower()
        
        if 'education' in title_lower:
            return 'education'
        elif 'experience' in title_lower or 'work' in title_lower:
            return 'experience'
        elif 'project' in title_lower:
            return 'projects'
        elif 'leadership' in title_lower:
            return 'leadership'
        elif 'skill' in title_lower:
            return 'skills'
        elif 'extracurricular' in title_lower or 'activities' in title_lower:
            return 'extracurricular'
        else:
            return title_lower.replace(' ', '_')
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract technical keywords from text."""
        keywords = set()
        
        # Technical keyword patterns
        patterns = [
            r'\b(?:Python|JavaScript|React|Node\.?js|SQL|MongoDB|PostgreSQL)\b',
            r'\b(?:Machine Learning|AI|LLM|API|Backend|Frontend|Full-?stack)\b',
            r'\b(?:Git|GitHub|AWS|Docker|Kubernetes|Linux|Unix)\b',
            r'\b(?:HTML|CSS|TypeScript|Java|C\+\+|Go|Rust)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.update(match.lower() for match in matches)
        
        return keywords
    
    def _score_sections_for_job(self, 
                               sections: List[ResumeSection], 
                               job_keywords: Set[str]) -> List[ResumeSection]:
        """Score resume sections based on relevance to job keywords."""
        for section in sections:
            # Base priority by section type
            base_priorities = {
                'experience': 1.0,
                'projects': 0.9,
                'skills': 0.8,
                'education': 0.7,
                'leadership': 0.5,
                'extracurricular': 0.3
            }
            
            base_score = base_priorities.get(section.name, 0.5)
            
            # Keyword matching bonus
            matching_keywords = section.keywords.intersection(job_keywords)
            keyword_bonus = len(matching_keywords) * 0.1
            
            # Length penalty for very verbose sections
            length_penalty = max(0, (len(section.content) - 2000) * 0.0001)
            
            section.priority_score = base_score + keyword_bonus - length_penalty
        
        return sorted(sections, key=lambda x: x.priority_score, reverse=True)
    
    def _rebuild_optimized_resume(self, 
                                 original_resume: str,
                                 scored_sections: List[ResumeSection],
                                 target_length: int,
                                 job_keywords: Set[str]) -> str:
        """Rebuild resume with optimized content."""
        # Start with document structure (header, packages, etc.)
        # Find the first section to separate header from content
        first_section_match = re.search(r'\\section\{', original_resume)
        
        if first_section_match:
            header = original_resume[:first_section_match.start()]
        else:
            # Fallback: include everything up to a reasonable point
            header = original_resume[:1000]
        
        # Footer (document end)
        footer_match = re.search(r'\\end{document}.*$', original_resume, re.DOTALL)
        footer = footer_match.group(0) if footer_match else "\n\\end{document}"
        
        # Ensure we always preserve essential LaTeX structure
        if not header.strip() or len(header) < 100:
            # Emergency fallback for missing header
            header = """\\documentclass[letterpaper,11pt]{article}
\\usepackage{latexsym}
\\usepackage[empty]{fullpage}
\\begin{document}
\\begin{center}
    \\textbf{\\scshape Resume}
\\end{center}
"""
        
        # Build optimized content
        optimized_sections = []
        current_length = len(header) + len(footer)
        
        # Always ensure we have some content, even if aggressive optimization
        if len(scored_sections) == 0:
            # If no sections found, preserve original content but truncated
            start_pos = first_section_match.start() if first_section_match else len(header)
            end_pos = len(original_resume) - len(footer)
            content_section = original_resume[start_pos:end_pos]
            
            if len(content_section) + current_length > target_length:
                content_section = content_section[:target_length - current_length - 100]  # Leave some buffer
            
            optimized_sections.append(content_section)
        else:
            for section in scored_sections:
                # Check if we have room for this section
                if current_length >= target_length * 0.9:
                    break
                    
                # Always include high-priority sections (experience, projects, skills)
                if section.name in ['experience', 'projects', 'skills']:
                    optimized_content = self._optimize_section_content(section, job_keywords)
                    optimized_sections.append(optimized_content)
                    current_length += len(optimized_content)
                
                # Include other sections if space allows
                elif current_length + len(section.content) * 0.7 < target_length:
                    optimized_content = self._optimize_section_content(section, job_keywords, aggressive=True)
                    optimized_sections.append(optimized_content)
                    current_length += len(optimized_content)
        
        # Combine all parts
        optimized_resume = header + "\n" + "\n".join(optimized_sections) + "\n" + footer
        
        return optimized_resume
    
    def _optimize_section_content(self, 
                                 section: ResumeSection, 
                                 job_keywords: Set[str],
                                 aggressive: bool = False) -> str:
        """Optimize individual section content."""
        if not aggressive:
            return section.content
        
        # For aggressive optimization, reduce verbosity while preserving keywords
        content = section.content
        
        # Shorten verbose bullet points
        content = re.sub(r'\\resumeItem\{([^}]{200,})\}', 
                        lambda m: self._shorten_bullet_point(m.group(1), job_keywords), 
                        content)
        
        return content
    
    def _shorten_bullet_point(self, bullet_content: str, job_keywords: Set[str]) -> str:
        """Intelligently shorten a bullet point while preserving key information."""
        # If the bullet point contains job keywords, preserve it more carefully
        has_keywords = any(keyword in bullet_content.lower() for keyword in job_keywords)
        
        if has_keywords and len(bullet_content) > 150:
            # Keep first sentence and any quantified results
            sentences = bullet_content.split(',')
            important_parts = []
            
            # Always keep first part
            if sentences:
                important_parts.append(sentences[0])
            
            # Keep parts with numbers/percentages
            for part in sentences[1:]:
                if re.search(r'\b\d+%|\b\d+x|\$\d+|\b\d+\+', part):
                    important_parts.append(part)
                    
            shortened = ', '.join(important_parts)
            return f"\\resumeItem{{{shortened}}}"
        
        elif not has_keywords and len(bullet_content) > 100:
            # More aggressive shortening for non-keyword content
            words = bullet_content.split()
            shortened = ' '.join(words[:20]) + ('...' if len(words) > 20 else '')
            return f"\\resumeItem{{{shortened}}}"
        
        return f"\\resumeItem{{{bullet_content}}}"
    
    def _extract_core_requirements(self, requirements: List[str]) -> List[str]:
        """Extract the most important requirements from the full list."""
        if not requirements:
            return []
        
        # Prioritize requirements with technical keywords
        tech_requirements = []
        other_requirements = []
        
        for req in requirements:
            if any(keyword in req.lower() for keyword in self.high_priority_keywords):
                tech_requirements.append(req)
            else:
                other_requirements.append(req)
        
        # Return top technical requirements + some others
        core_requirements = tech_requirements[:5] + other_requirements[:3]
        return core_requirements[:6]  # Max 6 requirements
    
    def _summarize_description(self, description: str, max_chars: int) -> str:
        """Summarize job description to essential points."""
        if len(description) <= max_chars:
            return description
        
        # Extract key sentences (those with technical keywords or requirements)
        sentences = re.split(r'[.!?]+', description)
        key_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in self.high_priority_keywords):
                key_sentences.append(sentence.strip())
            elif len(key_sentences) < 3 and ('experience' in sentence.lower() or 'required' in sentence.lower()):
                key_sentences.append(sentence.strip())
        
        # If we have key sentences, use them; otherwise truncate
        if key_sentences:
            summary = '. '.join(key_sentences) + '.'
            if len(summary) <= max_chars:
                return summary
        
        # Fallback: truncate at word boundary
        truncated = description[:max_chars]
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:
            truncated = truncated[:last_space]
        
        return truncated + '...'
    
    def _format_condensed_job_info(self, info: Dict) -> str:
        """Format condensed job information into a clean string."""
        formatted = f"""Job: {info['title']} at {info['company']}
Location: {info['location']}
Key Skills: {', '.join(info['key_skills'])}
Core Requirements: {'; '.join(info['core_requirements'])}
Summary: {info['summary']}"""
        
        return formatted

    def get_token_estimate(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimate: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def optimize_system_prompts(self, original_prompt: str) -> str:
        """Optimize system prompts by removing redundant instructions."""
        # Remove redundant phrases and overly verbose explanations
        optimizations = [
            (r'\n\s*\n\s*\n', '\n\n'),  # Multiple newlines
            (r'(?i)(critical|important|essential|vital):\s*', ''),  # Redundant emphasis
            (r'(?i)please note that ', ''),  # Redundant politeness
            (r'(?i)it is important to ', ''),  # Redundant emphasis
            (r'(?i)make sure to ', ''),  # Redundant instruction
            (r'(?i)remember to ', ''),  # Redundant instruction
        ]
        
        optimized = original_prompt
        for pattern, replacement in optimizations:
            optimized = re.sub(pattern, replacement, optimized)
        
        return optimized.strip()
