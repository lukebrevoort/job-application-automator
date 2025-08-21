#!/usr/bin/env python3
"""
Enhanced Fit Score Assessment Test

Tests the new Llama 3.2-powered fit assessment with detailed feedback.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ollama_integration import OllamaIntegration

# Create a simple JobPosting-like class for testing
class SimpleJobPosting:
    def __init__(self, title, company, location, description, skills, requirements, url="https://example.com"):
        self.title = title
        self.company = company
        self.location = location
        self.description = description
        self.skills = skills
        self.requirements = requirements
        self.url = url

def test_enhanced_fit_assessment():
    print("🚀 Testing Enhanced Fit Score Assessment with Llama 3.2")
    print("=" * 60)
    
    # Sample job posting
    job_posting = SimpleJobPosting(
        url="https://example.com/senior-python-dev",
        title="Senior Python Developer",
        company="TechStartup Inc",
        location="San Francisco, CA",
        description="""We are looking for a Senior Python Developer to join our fast-growing team. 
        You will be responsible for building scalable web applications using Django and FastAPI, 
        working with PostgreSQL databases, and deploying applications on AWS. 
        
        The ideal candidate has 5+ years of Python experience, strong knowledge of web frameworks,
        experience with cloud platforms, and excellent problem-solving skills. Knowledge of 
        machine learning libraries like TensorFlow or PyTorch is a plus.
        
        You will work closely with our product team to deliver high-quality features and 
        help mentor junior developers.""",
        skills=['Python', 'Django', 'FastAPI', 'PostgreSQL', 'AWS', 'REST APIs', 'Git', 'Machine Learning'],
        requirements=['5+ years Python experience', 'Web framework experience', 'Database knowledge', 'Cloud platform experience', 'Bachelor\'s degree preferred']
    )
    
    # Sample resume content (Luke's resume)
    resume_content = """
    Luke Brevoort
    720-862-5457 | luke@brevoort.com | luke.brevoort.com
    
    EDUCATION
    Stevens Institute of Technology, Hoboken, NJ
    Bachelor of Science in Computer Science, Machine Learning Concentration
    Expected: May 2027, GPA: 3.96
    Coursework: Data Structures and Algorithms, Machine Learning, HuggingFace Agentic AI
    
    EXPERIENCE
    Research Intern - Natural Language Processing Lab at Stevens (December 2024 -- Present)
    - Engineered large language model capabilities for interpreting complex multilingual text
    - Designing with Node.js, MongoDB Backend, Next.js Framework, and DeepSeek R1 Model
    
    PROJECTS
    Student Productivity Agent (February 2025 -- Present)
    - Designing an agentic system using LangGraph with specialized AI agents
    - Implemented using Canvas, Notion, and Calendar API connections
    
    Assignment Tracker (January 2025 -- February 2025)
    - Built assignment tool monitoring Canvas API and updating Notion database
    - Used Canvas RESTful API, Notion API, Python, and launchctl
    
    Personal Website (September 2024 -- December 2024)
    - Engineered responsive portfolio with on-device Llama 3.3 AI integration
    - Built with React/Next.js, achieved 95%+ Lighthouse performance score
    
    SKILLS
    Languages: Python, Go, JavaScript (React, Next.js), HTML/CSS, SQL, TypeScript
    Tools & Frameworks: Git, WebLLM, AWS, Node.js, Neural Networks, PostgreSQL, MySQL, LangGraph
    """
    
    print(f"📋 Job: {job_posting.title} at {job_posting.company}")
    print(f"🎯 Testing fit assessment with detailed feedback...")
    
    # Initialize Ollama client
    ollama_client = OllamaIntegration()
    
    try:
        # Assess fit score with enhanced system
        fit_score = ollama_client.assess_fit_score(
            resume_content=resume_content,
            job_posting=job_posting
        )
        
        print("\n" + "="*60)
        print("📊 FIT ASSESSMENT RESULTS")
        print("="*60)
        
        print(f"\n🎯 Overall Fit Score: {fit_score.fit_score}/100")
        print(f"📈 Confidence Level: {fit_score.confidence_level}")
        
        # Display score breakdown
        if hasattr(fit_score, 'score_breakdown') and fit_score.score_breakdown:
            print(f"\n📊 Score Breakdown:")
            for category, score in fit_score.score_breakdown.items():
                print(f"   {category.replace('_', ' ').title()}: {score}/100")
        
        # Display strengths
        print(f"\n💪 Key Strengths:")
        for strength in fit_score.strengths:
            print(f"   • {strength}")
        
        # Display weaknesses
        if fit_score.weaknesses:
            print(f"\n⚠️  Areas for Improvement:")
            for weakness in fit_score.weaknesses:
                print(f"   • {weakness}")
        
        # Display missing skills
        if fit_score.missing_skills:
            print(f"\n❌ Missing Skills:")
            for skill in fit_score.missing_skills:
                print(f"   • {skill}")
        
        # Display recommendations
        print(f"\n💡 Recommendations:")
        for rec in fit_score.recommendations:
            print(f"   • {rec}")
        
        # Display detailed feedback
        if hasattr(fit_score, 'detailed_feedback') and fit_score.detailed_feedback:
            print(f"\n📝 Detailed Analysis:")
            print("-" * 40)
            print(fit_score.detailed_feedback)
            print("-" * 40)
        
        print(f"\n✅ Enhanced fit assessment completed successfully!")
        print(f"🎉 The new system provides much more detailed and actionable feedback!")
        
    except Exception as e:
        print(f"\n❌ Error during fit assessment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_fit_assessment()
