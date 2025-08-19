"""
Ollama Integration Module

This module handles all interactions with Ollama-powered LLMs for:
1. Resume personalization based on job descriptions
2. Cover letter generation
3. Fit score assessment
4. Content analysis and optimization

The module uses carefully crafted prompts to ensure high-quality, professional output.
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import ollama
from job_scraper import JobPosting

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Want to use gpt-oss to experiment and use the model!
class ModelType(Enum):
    """Enum for different model types optimized for different tasks."""
    GENERAL = "gpt-oss:20b"           # Good general-purpose model
    CODE = "codellama"           # Better for technical content
    WRITING = "gpt-oss:20b"           # Good for writing tasks
    ANALYSIS = "gpt-oss:20b"          # Good for analysis tasks


@dataclass
class LLMResponse:
    """Container for LLM response data."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    response_time: float = 0.0
    success: bool = True
    error_message: str = ""


@dataclass
class ResumePersonalization:
    """Container for personalized resume content."""
    personalized_latex: str
    key_changes: List[str]
    skills_emphasized: List[str]
    removed_content: List[str]
    added_content: List[str]


@dataclass
class CoverLetterResult:
    """Container for generated cover letter content."""
    cover_letter_markdown: str
    key_points: List[str]
    tone: str
    personalization_level: str


@dataclass
class FitScoreAssessment:
    """Container for fit score analysis."""
    fit_score: int  # 1-100
    strengths: List[str]
    weaknesses: List[str]
    missing_skills: List[str]
    recommendations: List[str]
    confidence_level: str


class OllamaIntegration:
    """
    Main class for handling Ollama LLM operations.
    
    This class provides a high-level interface for all LLM-powered features
    in the job application automation system.
    """
    
    def __init__(self, 
                 default_model: str = "gpt-oss:20b",
                 timeout: int = 120,
                 max_retries: int = 3):
        """
        Initialize the Ollama integration.
        
        Args:
            default_model: Default model to use for operations
            timeout: Timeout for LLM requests in seconds
            max_retries: Maximum number of retry attempts
        """
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = ollama.Client()
        
        # Available models cache
        self._available_models = None
        
        # Verify Ollama connection
        self._verify_connection()
    
    def _verify_connection(self) -> bool:
        """Verify that Ollama is running and accessible."""
        try:
            response = self.client.list()
            models_list = []
            if 'models' in response:
                models_list = [m.get('name', m.get('model', 'unknown')) for m in response['models']]
            logger.info(f"Connected to Ollama. Available models: {models_list}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise ConnectionError(f"Cannot connect to Ollama service: {e}")
    
    def ensure_model(self, model_name: str) -> bool:
        """
        Ensure a specific model is available, pull if necessary.
        
        Args:
            model_name: Name of the model to ensure is available
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            # Check if model is already available
            response = self.client.list()
            available_models = []
            if 'models' in response:
                available_models = [m.get('name', m.get('model', 'unknown')) for m in response['models']]
            
            if model_name in available_models:
                logger.info(f"Model {model_name} is already available")
                return True
            
            # Try to pull the model
            logger.info(f"Pulling model {model_name}... This may take several minutes.")
            self.client.pull(model_name)
            logger.info(f"Successfully pulled model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ensure model {model_name}: {e}")
            return False
    
    def _generate_with_retry(self, 
                           model: str, 
                           prompt: str, 
                           system_prompt: str = None) -> LLMResponse:
        """
        Generate response with retry logic.
        
        Args:
            model: Model name to use
            prompt: User prompt
            system_prompt: System prompt (optional)
            
        Returns:
            LLMResponse object containing the result
        """
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat(
                    model=model,
                    messages=messages,
                    options={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 4000,  # Max tokens to generate
                    }
                )
                
                response_time = time.time() - start_time
                
                return LLMResponse(
                    content=response['message']['content'],
                    model=model,
                    response_time=response_time,
                    success=True
                )
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for model {model}: {e}")
                if attempt == self.max_retries - 1:
                    return LLMResponse(
                        content="",
                        model=model,
                        response_time=time.time() - start_time,
                        success=False,
                        error_message=str(e)
                    )
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def personalize_resume(self, 
                          base_latex_resume: str, 
                          job_posting: JobPosting) -> ResumePersonalization:
        """
        Personalize a LaTeX resume for a specific job posting.
        
        Args:
            base_latex_resume: The base LaTeX resume content
            job_posting: The job posting to personalize for
            
        Returns:
            ResumePersonalization object with the results
        """
        logger.info(f"Personalizing resume for {job_posting.title} at {job_posting.company}")
        
        system_prompt = """
        You are an expert resume writer and LaTeX specialist. Your job is to personalize a LaTeX resume 
        for a specific job posting while maintaining the original structure and formatting.
        
        Guidelines:
        1. Preserve ALL LaTeX formatting, commands, and structure
        2. Emphasize relevant experience and skills that match the job requirements
        3. Reorder bullet points to put most relevant items first
        4. Adjust descriptions to use keywords from the job posting
        5. Do NOT add fake experience or skills
        6. Do NOT change personal information, contact details, or education dates
        7. Focus on making existing content more relevant and impactful
        8. Use action verbs and quantified achievements where possible
        
        Return ONLY the modified LaTeX code, nothing else.
        """
        
        prompt = f"""
        Please personalize this LaTeX resume for the following job:
        
        Job Title: {job_posting.title}
        Company: {job_posting.company}
        Location: {job_posting.location}
        
        Job Description:
        {job_posting.description[:2000]}  # Truncate to avoid token limits
        
        Key Skills Required: {', '.join(job_posting.skills[:10])}
        Key Requirements: {'; '.join(job_posting.requirements[:8])}
        
        Base LaTeX Resume:
        {base_latex_resume}
        
        Personalize this resume to better match the job requirements. Focus on:
        1. Highlighting relevant experience
        2. Using keywords from the job description
        3. Emphasizing matching skills
        4. Reordering content for maximum impact
        """
        
        # Use general model for this task
        if not self.ensure_model(self.default_model):
            raise RuntimeError(f"Model {self.default_model} not available")
        
        response = self._generate_with_retry(
            model=self.default_model,
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        if not response.success:
            raise RuntimeError(f"Failed to personalize resume: {response.error_message}")
        
        # Extract key changes (simple heuristic)
        key_changes = self._extract_key_changes(base_latex_resume, response.content)
        
        return ResumePersonalization(
            personalized_latex=response.content,
            key_changes=key_changes,
            skills_emphasized=job_posting.skills[:5],
            removed_content=[],  # Could be enhanced to detect removed content
            added_content=[]     # Could be enhanced to detect added content
        )
    
    def generate_cover_letter(self, 
                            job_posting: JobPosting, 
                            resume_content: str = None,
                            personal_info: Dict[str, str] = None) -> CoverLetterResult:
        """
        Generate a personalized cover letter for a job posting.
        
        Args:
            job_posting: The job posting to write for
            resume_content: Optional resume content for context
            personal_info: Optional personal information dict
            
        Returns:
            CoverLetterResult object with the generated content
        """
        logger.info(f"Generating cover letter for {job_posting.title} at {job_posting.company}")
        
        system_prompt = """
        You are an expert career counselor and professional writer. Create compelling, 
        personalized cover letters that showcase ONLY the candidate's real experience and skills.
        
        CRITICAL RULES:
        1. NEVER invent, fabricate, or hallucinate experience that isn't in the resume
        2. ONLY use experiences, projects, skills, and achievements explicitly mentioned in the provided resume
        3. If the resume doesn't have direct experience for something, focus on transferable skills
        4. Use exact details from the resume (GPA, company names, project names, technologies)
        5. Do not exaggerate or embellish achievements beyond what's stated in the resume
        
        Guidelines:
        1. Use a professional, engaging tone
        2. Show genuine interest in the company and role  
        3. Highlight specific relevant experience and skills FROM THE RESUME ONLY
        4. Include quantified achievements ONLY if they exist in the resume
        5. Keep it concise (3-4 paragraphs)
        6. Use active voice and strong action verbs
        7. Customize for the specific company and role
        8. End with a strong call to action
        
        Format the output as clean markdown.
        """
        
        # Parse resume content to extract key information
        parsed_resume = self._parse_resume_content(resume_content) if resume_content else {}
        
        # Build context from available information
        context_info = f"Job Title: {job_posting.title}\\n"
        context_info += f"Company: {job_posting.company}\\n"
        if job_posting.location:
            context_info += f"Location: {job_posting.location}\\n"
        
        prompt = f"""
        Write a compelling cover letter for this job posting using ONLY the candidate's real experience from their resume.
        
        JOB POSTING:
        {context_info}
        
        Job Description:
        {job_posting.description[:2000]}
        
        Key Skills Required: {', '.join(job_posting.skills[:8])}
        Key Requirements: {'; '.join(job_posting.requirements[:6])}
        
        CANDIDATE'S ACTUAL RESUME CONTENT:
        {resume_content[:4000] if resume_content else "No resume provided"}
        
        PERSONAL INFORMATION:
        {str(personal_info) if personal_info else "No additional personal info"}
        
        INSTRUCTIONS:
        Write a professional cover letter that:
        1. Opens with enthusiasm for the specific role and company
        2. Highlights 2-3 most relevant qualifications/experiences FROM THE RESUME ONLY
        3. Shows understanding of the company's needs
        4. Uses specific projects, skills, and achievements mentioned in the resume
        5. Closes with a strong call to action
        
        CRITICAL: Do NOT invent any experience, projects, or achievements. Only use what's explicitly stated in the resume above.
        
        Format as clean markdown with proper structure.
        """
        
        response = self._generate_with_retry(
            model=self.default_model,
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        if not response.success:
            raise RuntimeError(f"Failed to generate cover letter: {response.error_message}")
        
        return CoverLetterResult(
            cover_letter_markdown=response.content,
            key_points=job_posting.skills[:3],
            tone="Professional and Enthusiastic",
            personalization_level="High"
        )
    
    def assess_fit_score(self, 
                        resume_content: str, 
                        job_posting: JobPosting,
                        cover_letter: str = None) -> FitScoreAssessment:
        """
        Assess the fit between a resume/cover letter and job posting.
        
        Args:
            resume_content: The resume content (LaTeX or plain text)
            job_posting: The job posting to assess against
            cover_letter: Optional cover letter content
            
        Returns:
            FitScoreAssessment object with detailed analysis
        """
        logger.info(f"Assessing fit score for {job_posting.title} at {job_posting.company}")
        
        system_prompt = """
        You are an experienced recruiter and hiring manager. Assess how well a candidate's 
        resume and cover letter match a job posting. Be honest and critical in your evaluation.
        
        Evaluation Criteria:
        1. Required skills match (40% weight)
        2. Experience level match (25% weight)
        3. Industry/domain knowledge (20% weight)
        4. Education requirements (10% weight)
        5. Soft skills and cultural fit (5% weight)
        
        Provide a realistic fit score from 1-100 where:
        - 90-100: Excellent fit, very likely to get interview
        - 80-89: Strong fit, good chance for interview
        - 70-79: Moderate fit, possible interview
        - 60-69: Weak fit, unlikely interview
        - Below 60: Poor fit, very unlikely interview
        
        Be critical and realistic. Don't inflate scores.
        """
        
        prompt = f"""
        Assess the fit between this candidate and job posting:
        
        JOB POSTING:
        Title: {job_posting.title}
        Company: {job_posting.company}
        Location: {job_posting.location}
        
        Description: {job_posting.description[:2000]}
        Required Skills: {', '.join(job_posting.skills)}
        Requirements: {'; '.join(job_posting.requirements)}
        
        CANDIDATE RESUME:
        {resume_content[:3000]}
        
        {"COVER LETTER:" + cover_letter[:1500] if cover_letter else ""}
        
        Provide your assessment in this JSON format:
        {{
            "fit_score": <number 1-100>,
            "strengths": ["strength1", "strength2", "strength3"],
            "weaknesses": ["weakness1", "weakness2"],
            "missing_skills": ["skill1", "skill2"],
            "recommendations": ["recommendation1", "recommendation2"],
            "confidence_level": "High|Medium|Low",
            "reasoning": "Brief explanation of the score"
        }}
        
        Be honest and critical. This is for self-improvement, not to impress.
        """
        
        response = self._generate_with_retry(
            model=self.default_model,
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        if not response.success:
            raise RuntimeError(f"Failed to assess fit score: {response.error_message}")
        
        try:
            # Try to parse JSON response
            result_data = json.loads(response.content.strip())
            
            return FitScoreAssessment(
                fit_score=result_data.get('fit_score', 0),
                strengths=result_data.get('strengths', []),
                weaknesses=result_data.get('weaknesses', []),
                missing_skills=result_data.get('missing_skills', []),
                recommendations=result_data.get('recommendations', []),
                confidence_level=result_data.get('confidence_level', 'Low')
            )
        except json.JSONDecodeError:
            # Fallback to simple parsing if JSON fails
            logger.warning("Failed to parse JSON response, using fallback parsing")
            
            # Simple regex-based score extraction
            import re
            score_match = re.search(r'(?:fit_score|score).*?(\d+)', response.content, re.IGNORECASE)
            fit_score = int(score_match.group(1)) if score_match else 50
            
            return FitScoreAssessment(
                fit_score=fit_score,
                strengths=["Analysis available in raw response"],
                weaknesses=["Could not parse detailed feedback"],
                missing_skills=[],
                recommendations=["Review LLM response manually"],
                confidence_level="Low"
            )
    
    def _extract_key_changes(self, original: str, modified: str) -> List[str]:
        """
        Extract key changes between original and modified content.
        This is a simple implementation - could be enhanced with diff libraries.
        """
        # Simple heuristic: look for different line counts and content
        orig_lines = len(original.split('\\n'))
        mod_lines = len(modified.split('\\n'))
        
        changes = []
        if abs(orig_lines - mod_lines) > 5:
            changes.append(f"Content length changed significantly")
        
        # Look for new keywords that weren't in original
        orig_lower = original.lower()
        mod_lower = modified.lower()
        
        common_keywords = ['python', 'javascript', 'react', 'aws', 'docker', 'kubernetes']
        for keyword in common_keywords:
            if keyword not in orig_lower and keyword in mod_lower:
                changes.append(f"Added emphasis on {keyword}")
        
        return changes[:5]  # Limit to 5 changes
    
    def _parse_resume_content(self, resume_content: str) -> Dict[str, Any]:
        """
        Parse resume content to extract structured information.
        This helps the LLM understand the candidate's actual background.
        
        Args:
            resume_content: The raw resume content (LaTeX or plain text)
            
        Returns:
            Dictionary with parsed resume information
        """
        if not resume_content:
            return {}
        
        # Clean LaTeX content for better parsing
        content = resume_content.replace('\\', '').replace('{', '').replace('}', '')
        content_lower = content.lower()
        
        parsed = {
            'education': [],
            'experience': [],
            'projects': [],
            'skills': [],
            'achievements': []
        }
        
        # Extract education (look for degree, GPA, university)
        if 'bachelor' in content_lower or 'stevens' in content_lower:
            parsed['education'].append("Bachelor of Science in Computer Science at Stevens Institute of Technology")
        if 'gpa' in content_lower:
            gpa_match = re.search(r'gpa[:\s]*(\d+\.\d+)', content_lower)
            if gpa_match:
                parsed['achievements'].append(f"GPA: {gpa_match.group(1)}")
        
        # Extract current role/experience
        if 'research intern' in content_lower:
            parsed['experience'].append("Research Intern at Natural Language Processing Lab")
        if 'student government' in content_lower:
            parsed['experience'].append("Assistant Vice President of Finance - Stevens Student Government")
        
        # Extract key skills
        skill_keywords = ['python', 'javascript', 'react', 'next.js', 'node.js', 'mongodb', 'git', 'aws']
        for skill in skill_keywords:
            if skill in content_lower:
                parsed['skills'].append(skill)
        
        # Extract notable achievements/projects
        if 'assignment tracker' in content_lower:
            parsed['projects'].append("Assignment Tracker - Canvas/Notion integration")
        if 'personal website' in content_lower:
            parsed['projects'].append("Personal Website with AI integration")
        if 'student productivity agent' in content_lower:
            parsed['projects'].append("FlowState - Student Productivity Agent using LangGraph")
        
        return parsed
    
    def list_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            models = self.client.list()
            return [m['name'] for m in models['models']]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def pull_recommended_models(self) -> bool:
        """Pull recommended models for the job application system."""
        recommended_models = ['llama2', 'codellama']
        
        logger.info("Pulling recommended models for job application automation...")
        
        success = True
        for model in recommended_models:
            if not self.ensure_model(model):
                logger.error(f"Failed to pull {model}")
                success = False
            else:
                logger.info(f"Successfully ensured {model} is available")
        
        return success


def main():
    """Example usage of the Ollama integration."""
    try:
        # Initialize the integration
        ollama_client = OllamaIntegration()
        
        # Check available models
        models = ollama_client.list_available_models()
        print(f"Available models: {models}")
        
        # Example job posting for testing
        from job_scraper import JobPosting
        test_job = JobPosting(
            url="https://example.com/job",
            title="Senior Python Developer",
            company="Tech Startup Inc.",
            location="San Francisco, CA",
            description="We are looking for a Senior Python Developer with experience in Django, React, and AWS.",
            skills=['python', 'django', 'react', 'aws'],
            requirements=['5+ years Python experience', 'Django framework knowledge', 'AWS cloud experience']
        )
        
        # Test cover letter generation
        print("\\nGenerating cover letter...")
        cover_letter = ollama_client.generate_cover_letter(test_job)
        print(f"Generated cover letter ({len(cover_letter.cover_letter_markdown)} chars)")
        
        # Test fit score assessment with dummy resume
        print("\\nAssessing fit score...")
        dummy_resume = """
        John Doe
        Python Developer with 6 years experience
        - Expert in Python, Django, Flask
        - Experience with AWS, Docker, Kubernetes  
        - Built web applications using React and JavaScript
        """
        
        fit_assessment = ollama_client.assess_fit_score(dummy_resume, test_job)
        print(f"Fit Score: {fit_assessment.fit_score}/100")
        print(f"Strengths: {fit_assessment.strengths}")
        
    except Exception as e:
        print(f"Error in Ollama integration: {e}")


if __name__ == "__main__":
    main()
