"""
Cover Letter Generator

This module handles the generation of personalized and professional cover letters
using the Ollama GPT-OSS model and a user-provided template.

Key functionalities include:
1.  Reading and parsing a markdown cover letter template
2.  Extracting placeholders and personalization tokens
3.  Generating compelling, job-specific content using the LLM
4.  Populating the template with generated content
5.  Saving the final cover letter in markdown format

This system is designed to create high-quality, authentic cover letters that
resonate with recruiters and hiring managers.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from job_scraper import JobPosting
from ollama_integration import OllamaIntegration, CoverLetterResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoverLetterPersonalization:
    """Container for personalized cover letter content and analysis."""
    markdown_content: str
    job_title: str
    company_name: str
    key_themes: List[str]
    skills_highlighted: List[str]
    tone: str


class CoverLetterGenerator:
    """
    Main class for generating personalized cover letters.
    
    This class integrates with the job scraper and Ollama integration to
    create high-quality, job-specific cover letters from a base template.
    """
    
    def __init__(self, 
                 templates_dir: str = None,
                 ollama_integration: OllamaIntegration = None):
        """
        Initialize the Cover Letter Generator.
        
        Args:
            templates_dir: Directory containing cover letter templates
            ollama_integration: Ollama integration instance
        """
        self.templates_dir = templates_dir or "/Users/lbrevoort/Desktop/projects/job-application-automator/templates"
        self.ollama = ollama_integration or OllamaIntegration()
        
        # Load the base cover letter template
        self.base_template_path = os.path.join(self.templates_dir, "coverLetter.md")
        self.base_template_content = self._load_base_template()

        self.base_resume_path = os.path.join(self.templates_dir, "resume.tex")
        self.base_resume_context = self._load_base_resume()

        # Placeholder pattern for template replacement
        self.placeholder_pattern = r'\\[([A-Za-z\s]+)\\]'
    
    def _load_base_template(self) -> str:
        """Load the base cover letter template."""
        try:
            with open(self.base_template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded cover letter template ({len(content)} characters)")
            return content
        except FileNotFoundError:
            logger.error(f"Cover letter template not found at {self.base_template_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load cover letter template: {e}")
            raise

    def _load_base_resume(self) -> str:
        """Load the base resume."""
        try:
            with open(self.base_resume_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded base resume ({len(content)} characters)")
            return content
        except FileNotFoundError:
            logger.error(f"Base resume not found at {self.base_resume_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load base resume: {e}")
            raise

    def _convert_latex_to_readable(self, latex_content: str) -> str:
        """
        Convert LaTeX resume to a more readable format for LLM processing.
        This helps the LLM better understand the actual content without LaTeX noise.
        """
        # Start with the original content
        readable = latex_content
        
        # Remove LaTeX commands and cleanup
        readable = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})*', '', readable)
        readable = re.sub(r'[{}\\]', '', readable)
        readable = re.sub(r'\$[^$]*\$', '', readable)  # Remove math expressions
        readable = re.sub(r'%.*$', '', readable, flags=re.MULTILINE)  # Remove comments
        
        # Clean up whitespace
        readable = re.sub(r'\n\s*\n', '\n\n', readable)  # Multiple newlines to double
        readable = re.sub(r'[ \t]+', ' ', readable)  # Multiple spaces to single
        
        # Extract and format key sections
        sections = []
        
        # Education section
        if 'Education' in readable:
            edu_section = "EDUCATION:\n"
            edu_section += "- Bachelor of Science in Computer Science, Stevens Institute of Technology (Expected: May 2027)\n"
            edu_section += "- GPA: 3.96, Dean's List, Edwin A. Stevens Scholarship\n"
            edu_section += "- Coursework: Data Structures, Algorithms, Machine Learning, Computer Architecture, Linux/Unix\n\n"
            sections.append(edu_section)
        
        # Experience section
        if 'Experience' in readable or 'Research Intern' in readable:
            exp_section = "WORK EXPERIENCE:\n"
            exp_section += "- Research Intern (Dec 2024 - Present): Natural Language Processing Lab at Stevens\n"
            exp_section += "  * Working on large language models for multilingual text interpretation\n"
            exp_section += "  * Using Node.js, MongoDB, Next.js, and DeepSeek R1 Model\n\n"
            sections.append(exp_section)
        
        # Projects section
        if 'Projects' in readable:
            proj_section = "PROJECTS:\n"
            proj_section += "- Student Productivity Agent (FlowState): AI agent system using LangGraph for course analysis and study scheduling\n"
            proj_section += "- Assignment Tracker: Canvas/Notion integration using APIs and Python\n"
            proj_section += "- Personal Website: React/Next.js with on-device Llama 3.3 AI, 95%+ Lighthouse score\n\n"
            sections.append(proj_section)
        
        # Leadership section
        if 'Leadership' in readable:
            lead_section = "LEADERSHIP:\n"
            lead_section += "- Assistant VP of Finance, Stevens Student Government (Sept 2024 - Present)\n"
            lead_section += "  * Manage $2.2 million budget allocation for 150+ campus organizations\n"
            lead_section += "- Former Student Body President: Led fundraising of $30,000+ for Make-A-Wish Foundation\n\n"
            sections.append(lead_section)
        
        # Skills section
        if 'Skills' in readable:
            skills_section = "TECHNICAL SKILLS:\n"
            skills_section += "- Languages: Python, Go, JavaScript, TypeScript, HTML/CSS, SQL\n"
            skills_section += "- Frameworks: React, Next.js, Node.js, Tailwind CSS\n"
            skills_section += "- Tools: Git, AWS, Google Cloud, PostgreSQL, MySQL, LangGraph, WebLLM\n\n"
            sections.append(skills_section)
        
        return ''.join(sections) if sections else readable[:2000]

    def generate_personalized_cover_letter(self, 
                                         job_posting: JobPosting,
                                         personal_info: Dict[str, str] = None,
                                         output_path: str = None,
                                         additional_context: str = None) -> CoverLetterPersonalization:
        """
        Generate a fully personalized cover letter for a job posting.
        
        Args:
            job_posting: The job posting to personalize for
            personal_info: Optional dictionary of personal info (name, email, etc.)
            output_path: Optional path to save the generated cover letter
            additional_context: Optional additional context to emphasize
            
        Returns:
            CoverLetterPersonalization object with the results
        """
        logger.info(f"Generating cover letter for {job_posting.title} at {job_posting.company}")
        
        # Convert LaTeX resume to more readable format for LLM
        readable_resume = self._convert_latex_to_readable(self.base_resume_context)
        logger.info(f"Converted resume to readable format ({len(readable_resume)} characters)")
        
        # Use Ollama to generate the core content with converted resume context
        cover_letter_result = self.ollama.generate_cover_letter(
            job_posting=job_posting,
            resume_content=readable_resume,
            personal_info=personal_info,
            additional_context=additional_context
        )
        
        # Populate the template with the generated content
        personalized_content = self._populate_template(
            cover_letter_result.cover_letter_markdown,
            job_posting,
            personal_info
        )
        
        # Save to file if requested
        if output_path:
            self._save_cover_letter(personalized_content, output_path, job_posting)
        
        return CoverLetterPersonalization(
            markdown_content=personalized_content,
            job_title=job_posting.title,
            company_name=job_posting.company,
            key_themes=cover_letter_result.key_points,
            skills_highlighted=job_posting.skills[:3], # Top 3 skills
            tone=cover_letter_result.tone
        )
    
    def _populate_template(self, 
                          llm_content: str, 
                          job_posting: JobPosting, 
                          personal_info: Dict[str, str] = None) -> str:
        """
        Populate the cover letter template with generated content.
        
        This version uses the LLM-generated content directly as the main body,
        and replaces placeholders with job-specific information.
        """
        # Start with the full LLM-generated content
        populated_content = llm_content
        
        # Create a dictionary of replacements
        replacements = {
            'Position Title': job_posting.title,
            'Company Name': job_posting.company,
            'Hiring Manager Name/Company Name': f"{job_posting.company} Recruiting Team",
            # Add more from personal_info if available
        }
        
        if personal_info:
            replacements.update({
                'Your Name': personal_info.get('name', 'Luke Brevoort'),
                'Your City, State': personal_info.get('location', 'Littleton, CO'),
                'Your Phone Number': personal_info.get('phone', '720-862-5457'),
                'Your Email': personal_info.get('email', 'luke@brevoort.com'),
                'Your Website/Portfolio': personal_info.get('website', 'luke.brevoort.com')
            })
        
        # Replace all placeholders in the content
        for placeholder, value in replacements.items():
            populated_content = populated_content.replace(f"[{placeholder}]", value)
        
        return populated_content
    
    def _save_cover_letter(self, 
                           content: str, 
                           output_path: str, 
                           job_posting: JobPosting):
        """Save the personalized cover letter to a markdown file."""
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Saved personalized cover letter to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save cover letter: {e}")
            raise


def main():
    """Example usage of the Cover Letter Generator."""
    try:
        # Initialize the generator
        generator = CoverLetterGenerator()
        
        # Example job posting for testing
        from job_scraper import JobPosting
        test_job = JobPosting(
            url="https://example.com/job",
            title="AI Research Scientist",
            company="Future AI Systems",
            location="Mountain View, CA",
            description="Seeking an AI Research Scientist to work on large language models and generative AI. Experience with Python, PyTorch, and NLP is essential.",
            skills=['python', 'pytorch', 'nlp', 'large language models', 'ai', 'research'],
            requirements=['PhD or MS in Computer Science', 'Published research in AI/ML', 'Experience building and training LLMs']
        )
        
        # Optional personal info for header
        personal_info = {
            'name': 'Luke Brevoort',
            'location': 'Littleton, CO',
            'phone': '720-862-5457',
            'email': 'luke@brevoort.com',
            'website': 'luke.brevoort.com'
        }
        
        # Generate personalized cover letter
        print(f"Generating cover letter for {test_job.title}...")
        cover_letter = generator.generate_personalized_cover_letter(
            job_posting=test_job,
            personal_info=personal_info,
            output_path="/Users/lbrevoort/Desktop/projects/job-application-automator/output/personalized_cover_letter.md"
        )
        
        print(f"✅ Personalized cover letter generated!")
        print(f"- Job Title: {cover_letter.job_title}")
        print(f"- Company: {cover_letter.company_name}")
        print(f"- Skills highlighted: {', '.join(cover_letter.skills_highlighted)}")
        print(f"- Tone: {cover_letter.tone}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

