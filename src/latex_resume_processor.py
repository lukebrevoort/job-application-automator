"""
LaTeX Resume Processor

This module handles intelligent processing and personalization of LaTeX resumes
based on job requirements. It uses the Ollama GPT-OSS model to:

1. Parse and understand the structure of LaTeX resumes
2. Identify key sections and content areas
3. Personalize content to match job requirements
4. Preserve LaTeX formatting and structure
5. Generate personalized versions while maintaining professional quality

The processor is specifically designed to work with Luke Brevoort's resume template
but can be adapted for other LaTeX resume formats.
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from job_scraper import JobPosting
from ollama_integration import OllamaIntegration, ResumePersonalization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResumeSection:
    """Represents a section of the resume with its content."""
    name: str
    start_line: int
    end_line: int
    content: List[str]
    section_type: str  # 'education', 'experience', 'projects', 'skills', etc.


@dataclass
class ResumeAnalysis:
    """Analysis results of a resume structure."""
    sections: List[ResumeSection]
    skills: List[str]
    experience_keywords: List[str]
    projects: List[str]
    total_lines: int


@dataclass
class PersonalizationStrategy:
    """Strategy for personalizing resume content."""
    skills_to_emphasize: List[str]
    experience_focus: List[str]
    project_priorities: List[str]
    keywords_to_add: List[str]
    sections_to_reorder: List[str]


class LaTeXResumeProcessor:
    """
    Advanced LaTeX resume processor for intelligent job-specific personalization.
    
    This class provides comprehensive resume processing capabilities including:
    - Structure analysis and parsing
    - Content extraction and categorization  
    - Intelligent personalization based on job requirements
    - LaTeX formatting preservation
    - Quality validation
    """
    
    def __init__(self, 
                 templates_dir: str = None,
                 ollama_integration: OllamaIntegration = None):
        """
        Initialize the LaTeX Resume Processor.
        
        Args:
            templates_dir: Directory containing resume templates
            ollama_integration: Ollama integration instance
        """
        self.templates_dir = templates_dir or "/Users/lbrevoort/Desktop/projects/job-application-automator/templates"
        self.ollama = ollama_integration or OllamaIntegration()
        
        # Load the base resume template
        self.base_resume_path = os.path.join(self.templates_dir, "resume.tex")
        self.base_resume_content = self._load_base_resume()
        
        # Common LaTeX commands and patterns
        self.latex_commands = {
            'section': r'\\section\{([^}]+)\}',
            'resumeSubheading': r'\\resumeSubheading\s*\{([^}]+)\}\{([^}]+)\}\s*\{([^}]+)\}\{([^}]+)\}',
            'resumeProjectHeading': r'\\resumeProjectHeading\s*\{([^}]+)\}\{([^}]+)\}',
            'resumeItem': r'\\resumeItem\{([^}]+)\}',
            'textbf': r'\\textbf\{([^}]+)\}',
        }
        
        # Section mappings and priorities
        self.section_priorities = {
            'education': 1,
            'work experience': 2, 
            'experience': 2,
            'projects': 3,
            'leadership experience': 4,
            'skills': 5,
            'extracurricular activities': 6
        }
    
    def _load_base_resume(self) -> str:
        """Load the base resume template."""
        try:
            with open(self.base_resume_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded base resume template ({len(content)} characters)")
            return content
        except FileNotFoundError:
            logger.error(f"Base resume template not found at {self.base_resume_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load base resume: {e}")
            raise
    
    def analyze_resume_structure(self, resume_content: str = None) -> ResumeAnalysis:
        """
        Analyze the structure and content of a LaTeX resume.
        
        Args:
            resume_content: Resume content to analyze (defaults to base template)
            
        Returns:
            ResumeAnalysis object with detailed structure information
        """
        content = resume_content or self.base_resume_content
        lines = content.split('\n')
        
        sections = []
        skills = []
        experience_keywords = []
        projects = []
        
        current_section = None
        section_content = []
        
        for i, line in enumerate(lines):
            # Look for section headers
            section_match = re.search(self.latex_commands['section'], line)
            if section_match:
                # Save previous section
                if current_section:
                    sections.append(ResumeSection(
                        name=current_section,
                        start_line=section_start,
                        end_line=i-1,
                        content=section_content,
                        section_type=self._classify_section_type(current_section)
                    ))
                
                # Start new section
                current_section = section_match.group(1).lower()
                section_start = i
                section_content = [line]
            elif current_section:
                section_content.append(line)
                
                # Extract skills from Skills section
                if 'skill' in current_section.lower():
                    skills.extend(self._extract_skills_from_line(line))
                
                # Extract experience keywords
                if any(term in current_section.lower() for term in ['experience', 'work']):
                    experience_keywords.extend(self._extract_experience_keywords(line))
                
                # Extract project names
                if 'project' in current_section.lower():
                    project_name = self._extract_project_name(line)
                    if project_name:
                        projects.append(project_name)
        
        # Add final section
        if current_section:
            sections.append(ResumeSection(
                name=current_section,
                start_line=section_start,
                end_line=len(lines)-1,
                content=section_content,
                section_type=self._classify_section_type(current_section)
            ))
        
        return ResumeAnalysis(
            sections=sections,
            skills=list(set(skills)),
            experience_keywords=list(set(experience_keywords)),
            projects=projects,
            total_lines=len(lines)
        )
    
    def _classify_section_type(self, section_name: str) -> str:
        """Classify the type of resume section."""
        name_lower = section_name.lower()
        
        if 'education' in name_lower:
            return 'education'
        elif any(term in name_lower for term in ['experience', 'work', 'employment']):
            return 'experience'
        elif 'project' in name_lower:
            return 'projects'
        elif any(term in name_lower for term in ['skill', 'technical', 'programming']):
            return 'skills'
        elif 'leadership' in name_lower:
            return 'leadership'
        elif any(term in name_lower for term in ['extracurricular', 'activities', 'involvement']):
            return 'activities'
        else:
            return 'other'
    
    def _extract_skills_from_line(self, line: str) -> List[str]:
        """Extract technical skills from a line of text."""
        skills = []
        
        # Look for skill patterns in LaTeX format
        # Pattern: \textbf{Languages}{: Python, Go, JavaScript...}
        skills_match = re.search(r'\\textbf\{[^}]*\}\{:\s*([^}]+)\}', line)
        if skills_match:
            skill_text = skills_match.group(1)
            # Split by commas and clean up
            raw_skills = [s.strip() for s in skill_text.split(',')]
            for skill in raw_skills:
                # Remove parenthetical information
                clean_skill = re.sub(r'\([^)]*\)', '', skill).strip()
                if clean_skill:
                    skills.append(clean_skill.lower())
        
        return skills
    
    def _extract_experience_keywords(self, line: str) -> List[str]:
        """Extract experience-related keywords from a line."""
        keywords = []
        
        # Common technical terms to extract
        tech_patterns = [
            r'\b(python|javascript|react|node\.?js|aws|docker|kubernetes|sql|mongodb|postgresql)\b',
            r'\b(machine learning|ai|artificial intelligence|neural networks|data science)\b',
            r'\b(api|rest|graphql|microservices|cloud|devops|ci/cd)\b',
            r'\b(git|github|linux|unix|agile|scrum)\b'
        ]
        
        line_lower = line.lower()
        for pattern in tech_patterns:
            matches = re.findall(pattern, line_lower, re.IGNORECASE)
            keywords.extend(matches)
        
        return keywords
    
    def _extract_project_name(self, line: str) -> Optional[str]:
        """Extract project name from a LaTeX project heading."""
        project_match = re.search(self.latex_commands['resumeProjectHeading'], line)
        if project_match:
            # Extract project name from \textbf{Project Name}
            project_text = project_match.group(1)
            name_match = re.search(r'\\textbf\{([^}]+)\}', project_text)
            if name_match:
                return name_match.group(1).strip()
        
        return None
    
    def create_personalization_strategy(self, 
                                      job_posting: JobPosting, 
                                      resume_analysis: ResumeAnalysis) -> PersonalizationStrategy:
        """
        Create a strategy for personalizing the resume based on job requirements.
        
        Args:
            job_posting: The job posting to tailor for
            resume_analysis: Analysis of the current resume
            
        Returns:
            PersonalizationStrategy with specific recommendations
        """
        # Match job skills with resume skills
        job_skills = [skill.lower() for skill in job_posting.skills]
        resume_skills = [skill.lower() for skill in resume_analysis.skills]
        
        # Find matching skills to emphasize
        skills_to_emphasize = []
        for job_skill in job_skills:
            for resume_skill in resume_skills:
                if job_skill in resume_skill or resume_skill in job_skill:
                    skills_to_emphasize.append(resume_skill)
        
        # Extract key experience focuses from job description
        experience_focus = self._extract_experience_focus(job_posting.description)
        
        # Determine project priorities based on job requirements
        project_priorities = self._prioritize_projects(job_posting, resume_analysis.projects)
        
        # Generate keywords to add based on job posting
        keywords_to_add = self._generate_keywords_to_add(job_posting, resume_analysis)
        
        return PersonalizationStrategy(
            skills_to_emphasize=skills_to_emphasize,
            experience_focus=experience_focus,
            project_priorities=project_priorities,
            keywords_to_add=keywords_to_add,
            sections_to_reorder=self._determine_section_order(job_posting)
        )
    
    def _extract_experience_focus(self, job_description: str) -> List[str]:
        """Extract key experience areas to focus on from job description."""
        focus_areas = []
        
        # Handle both string and list inputs from token optimization
        if isinstance(job_description, list):
            description_text = ' '.join(job_description)
        else:
            description_text = job_description
        
        description_lower = description_text.lower()
        
        # Look for key focus patterns
        focus_patterns = {
            'backend development': ['backend', 'server-side', 'api development'],
            'frontend development': ['frontend', 'react', 'user interface', 'ui/ux'],
            'full-stack development': ['full-stack', 'full stack', 'end-to-end'],
            'machine learning': ['machine learning', 'ml', 'artificial intelligence', 'ai'],
            'data engineering': ['data pipeline', 'etl', 'data processing'],
            'cloud computing': ['aws', 'cloud', 'kubernetes', 'docker'],
            'leadership': ['lead', 'mentor', 'team', 'manage']
        }
        
        for focus, keywords in focus_patterns.items():
            if any(keyword in description_lower for keyword in keywords):
                focus_areas.append(focus)
        
        return focus_areas
    
    def _prioritize_projects(self, job_posting: JobPosting, projects: List[str]) -> List[str]:
        """Prioritize projects based on relevance to job posting."""
        job_text = (str(job_posting.description) + ' ' + ' '.join(job_posting.skills)).lower()        
        project_scores = {}
        for project in projects:
            score = 0
            project_lower = project.lower()
            
            # Score based on keyword matches
            if any(tech in job_text for tech in ['ai', 'machine learning', 'llm']) and \
               any(tech in project_lower for tech in ['ai', 'agent', 'llm', 'productivity']):
                score += 3
            
            if 'web' in job_text and any(web in project_lower for web in ['website', 'web', 'react', 'next.js']):
                score += 2
            
            if 'api' in job_text and 'api' in project_lower:
                score += 2
            
            if any(skill.lower() in project_lower for skill in job_posting.skills):
                score += 1
            
            project_scores[project] = score
        
        # Sort by score descending
        return sorted(projects, key=lambda p: project_scores.get(p, 0), reverse=True)
    
    def _generate_keywords_to_add(self, job_posting: JobPosting, resume_analysis: ResumeAnalysis) -> List[str]:
        """Generate keywords that should be added to the resume."""
        keywords_to_add = []
        
        # Find job skills not prominently featured in resume
        job_skills = [skill.lower() for skill in job_posting.skills]
        resume_skills = [skill.lower() for skill in resume_analysis.skills]
        
        for job_skill in job_skills:
            if not any(job_skill in resume_skill for resume_skill in resume_skills):
                keywords_to_add.append(job_skill)
        
        return keywords_to_add[:5]  # Limit to top 5
    
    def _determine_section_order(self, job_posting: JobPosting) -> List[str]:
        """Determine optimal section ordering based on job requirements."""
        base_order = ['education', 'experience', 'projects', 'leadership', 'skills', 'activities']
        
        # Adjust based on job focus
        job_text = job_posting.description.lower()
        
        # If heavily technical, promote projects
        if any(term in job_text for term in ['technical', 'programming', 'development', 'engineering']):
            if 'projects' in base_order:
                base_order.remove('projects')
                base_order.insert(1, 'projects')  # After education
        
        # If leadership role, promote leadership experience
        if any(term in job_text for term in ['lead', 'senior', 'principal', 'manager']):
            if 'leadership' in base_order:
                base_order.remove('leadership')
                base_order.insert(2, 'leadership')  # After experience
        
        return base_order
    
    def personalize_resume(self, 
                          job_posting: JobPosting,
                          output_path: str = None,
                          strategy: PersonalizationStrategy = None,
                          additional_context: str = None) -> ResumePersonalization:
        """
        Create a personalized version of the resume for a specific job posting.
        
        Args:
            job_posting: The job posting to personalize for
            output_path: Optional path to save the personalized resume
            strategy: Optional custom personalization strategy
            additional_context: Optional additional context to provide to the LLM
            
        Returns:
            ResumePersonalization object with results
        """
        logger.info(f"Personalizing resume for {job_posting.title} at {job_posting.company}")
        
        # Analyze current resume
        analysis = self.analyze_resume_structure()
        
        # Create or use personalization strategy
        if not strategy:
            strategy = self.create_personalization_strategy(job_posting, analysis)
        
        # Use Ollama to generate personalized content
        personalization = self.ollama.personalize_resume(
            self.base_resume_content, 
            job_posting,
            additional_context=additional_context
        )
        
        # Apply additional processing based on strategy
        enhanced_latex = self._apply_personalization_strategy(
            personalization.personalized_latex, 
            strategy, 
            job_posting
        )
        
        # Validate LaTeX quality
        validation_results = self._validate_latex_quality(enhanced_latex)
        
        # Save to file if requested
        if output_path:
            self._save_personalized_resume(enhanced_latex, output_path, job_posting)
        
        # Update personalization results
        personalization.personalized_latex = enhanced_latex
        personalization.key_changes.extend([
            f"Emphasized skills: {', '.join(strategy.skills_to_emphasize[:3])}",
            f"Added keywords: {', '.join(strategy.keywords_to_add[:3])}",
            f"Prioritized projects: {', '.join(strategy.project_priorities[:2])}"
        ])
        
        return personalization
    
    def _apply_personalization_strategy(self, 
                                      latex_content: str, 
                                      strategy: PersonalizationStrategy,
                                      job_posting: JobPosting) -> str:
        """Apply additional personalization based on strategy."""
        
        # This is a simplified version - the main personalization happens in Ollama
        # Here we could add additional fine-tuning if needed
        
        enhanced_content = latex_content
        
        # Ensure key skills are properly formatted and emphasized
        for skill in strategy.skills_to_emphasize:
            # Look for skill mentions and make sure they're emphasized
            skill_pattern = re.compile(rf'\b{re.escape(skill)}\b', re.IGNORECASE)
            if skill_pattern.search(enhanced_content) and '\\textbf{' not in skill:
                # This is a simplified enhancement - in practice, you'd want more sophisticated logic
                pass
        
        return enhanced_content
    
    def _validate_latex_quality(self, latex_content: str) -> Dict[str, Any]:
        """Validate the quality and correctness of generated LaTeX."""
        validation = {
            'valid_structure': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for balanced braces
        open_braces = latex_content.count('{')
        close_braces = latex_content.count('}')
        if open_braces != close_braces:
            validation['valid_structure'] = False
            validation['issues'].append(f"Unbalanced braces: {open_braces} open, {close_braces} close")
        
        # Check for required sections
        required_sections = ['\\section{Education}', '\\section{Work Experience}', '\\section{Skills}']
        for section in required_sections:
            if section not in latex_content:
                validation['warnings'].append(f"Missing section: {section}")
        
        # Check for document structure
        if '\\begin{document}' not in latex_content:
            validation['valid_structure'] = False
            validation['issues'].append("Missing \\begin{document}")
        
        if '\\end{document}' not in latex_content:
            validation['valid_structure'] = False
            validation['issues'].append("Missing \\end{document}")
        
        return validation
    
    def _save_personalized_resume(self, 
                                latex_content: str, 
                                output_path: str, 
                                job_posting: JobPosting):
        """Save the personalized resume to a file."""
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            # Also save metadata
            metadata_path = output_path.replace('.tex', '_metadata.json')
            metadata = {
                'job_title': job_posting.title,
                'company': job_posting.company,
                'url': job_posting.url,
                'skills_required': job_posting.skills,
                'personalization_date': str(datetime.now()),
                'template_used': 'resume.tex'
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved personalized resume to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save personalized resume: {e}")
            raise
    
    def get_resume_stats(self) -> Dict[str, Any]:
        """Get statistics about the base resume."""
        analysis = self.analyze_resume_structure()
        
        return {
            'total_lines': analysis.total_lines,
            'sections_count': len(analysis.sections),
            'sections': [section.name for section in analysis.sections],
            'skills_count': len(analysis.skills),
            'skills': analysis.skills,
            'projects_count': len(analysis.projects),
            'projects': analysis.projects,
            'experience_keywords': analysis.experience_keywords
        }


def main():
    """Example usage of the LaTeX Resume Processor."""
    try:
        # Initialize the processor
        processor = LaTeXResumeProcessor()
        
        # Get resume statistics
        stats = processor.get_resume_stats()
        print(f"Resume Statistics:")
        print(f"- Total lines: {stats['total_lines']}")
        print(f"- Sections: {', '.join(stats['sections'])}")
        print(f"- Skills found: {len(stats['skills'])} ({', '.join(stats['skills'][:5])}...)")
        print(f"- Projects: {', '.join(stats['projects'])}")
        
        # Example job posting for testing
        from job_scraper import JobPosting
        test_job = JobPosting(
            url="https://example.com/job",
            title="Machine Learning Engineer",
            company="Tech Innovation Corp",
            location="San Francisco, CA",
            description="We are seeking a Machine Learning Engineer with experience in Python, TensorFlow, and AWS. The role involves building AI agents and working with large language models.",
            skills=['python', 'tensorflow', 'aws', 'machine learning', 'ai', 'pytorch'],
            requirements=['MS in Computer Science or related field', 'Experience with ML frameworks', 'Python programming']
        )
        
        # Create personalization strategy
        analysis = processor.analyze_resume_structure()
        strategy = processor.create_personalization_strategy(test_job, analysis)
        
        print(f"\nPersonalization Strategy for {test_job.title}:")
        print(f"- Skills to emphasize: {', '.join(strategy.skills_to_emphasize)}")
        print(f"- Experience focus: {', '.join(strategy.experience_focus)}")
        print(f"- Project priorities: {', '.join(strategy.project_priorities)}")
        print(f"- Keywords to add: {', '.join(strategy.keywords_to_add)}")
        
        # Generate personalized resume
        print(f"\nGenerating personalized resume...")
        personalization = processor.personalize_resume(
            test_job,
            output_path="/Users/lbrevoort/Desktop/projects/job-application-automator/output/personalized_resume.tex"
        )
        
        print(f"✅ Personalized resume generated!")
        print(f"Key changes: {', '.join(personalization.key_changes)}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
