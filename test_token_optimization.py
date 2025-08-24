#!/usr/bin/env python3
"""
Token Optimization Test Script

This script tests the token optimization functionality by comparing
the original and optimized token usage across different scenarios.
"""

import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ollama_integration import OllamaIntegration
from token_optimizer import TokenOptimizer
from config import SystemConfig
from job_scraper import JobPosting, RawJobContent

def create_test_job_posting():
    """Create a comprehensive test job posting."""
    return JobPosting(
        url="https://example.com/test-job",
        title="Senior Full Stack Software Engineer",
        company="TechCorp Inc.",
        location="San Francisco, CA",
        description="""
        We are seeking an experienced Senior Full Stack Software Engineer to join our growing team. 
        In this role, you will be responsible for designing, developing, and maintaining scalable web applications 
        using modern technologies including React, Node.js, Python, and AWS cloud services.
        
        You will work closely with cross-functional teams including product managers, designers, and other engineers 
        to deliver high-quality software solutions that meet business requirements. The ideal candidate should have 
        strong experience in both frontend and backend development, with a passion for clean code, testing, and 
        continuous improvement.
        
        Our technology stack includes React for frontend development, Node.js and Python for backend services, 
        PostgreSQL and MongoDB for data storage, Docker for containerization, and AWS for cloud infrastructure. 
        We practice agile development methodologies and maintain high code quality standards through comprehensive 
        testing and code review processes.
        
        This is an excellent opportunity to work on challenging technical problems while contributing to a product 
        that impacts thousands of users. We offer competitive compensation, comprehensive benefits, and a 
        collaborative work environment that encourages professional growth and innovation.
        """,
        skills=[
            "React", "Node.js", "Python", "JavaScript", "TypeScript", "HTML", "CSS", 
            "PostgreSQL", "MongoDB", "SQL", "AWS", "Docker", "Git", "REST APIs",
            "GraphQL", "Redis", "Elasticsearch", "Kubernetes", "CI/CD", "Linux"
        ],
        requirements=[
            "5+ years of professional software development experience",
            "Strong proficiency in React and modern JavaScript/TypeScript",
            "Experience with Node.js and Python for backend development",
            "Knowledge of relational and NoSQL databases (PostgreSQL, MongoDB)",
            "Experience with cloud platforms, preferably AWS",
            "Understanding of containerization technologies (Docker, Kubernetes)",
            "Familiarity with CI/CD pipelines and DevOps practices",
            "Strong problem-solving skills and attention to detail",
            "Excellent communication and collaboration abilities",
            "Bachelor's degree in Computer Science or related field"
        ],
        job_type="Full-time",
        experience_level="Senior",
        salary_range="$140,000 - $180,000"
    )

def read_test_resume():
    """Read the test resume template."""
    resume_path = os.path.join(os.path.dirname(__file__), 'templates', 'resume.tex')
    try:
        with open(resume_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Resume template not found at {resume_path}")
        return None

def test_token_optimization():
    """Test token optimization across different configurations."""
    print("=" * 70)
    print("  TOKEN OPTIMIZATION TEST")
    print("=" * 70)
    
    # Test configurations
    configs = {
        "Default": SystemConfig.default(),
        "Memory Optimized": SystemConfig.memory_optimized(),
        "High Quality": SystemConfig.high_quality()
    }
    
    # Create test data
    job_posting = create_test_job_posting()
    resume_content = read_test_resume()
    
    if not resume_content:
        print("❌ Could not load resume template for testing")
        return
    
    print(f"📄 Test Resume Length: {len(resume_content):,} characters")
    print(f"🏢 Test Job Description Length: {len(job_posting.description):,} characters")
    print(f"🔧 Test Skills Count: {len(job_posting.skills)}")
    print(f"📋 Test Requirements Count: {len(job_posting.requirements)}")
    print()
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"Testing {config_name} Configuration...")
        print("-" * 50)
        
        # Initialize optimizer
        optimizer = TokenOptimizer()
        
        # Test resume optimization
        if config.token_optimization.enabled:
            optimized_resume = optimizer.optimize_resume_for_job(
                resume_content, job_posting, 
                max_tokens=config.token_optimization.max_resume_tokens
            )
            
            optimized_job = optimizer.condense_job_posting(
                job_posting, 
                max_tokens=config.token_optimization.max_job_description_tokens
            )
            
            resume_compression = optimized_resume.compression_ratio
            job_compression = optimized_job.compression_ratio
            
            print(f"📄 Resume: {len(resume_content):,} → {len(optimized_resume.optimized_text):,} chars "
                  f"({resume_compression:.2%} of original)")
            print(f"🏢 Job Description: {len(job_posting.description):,} → {len(optimized_job.optimized_text):,} chars "
                  f"({job_compression:.2%} of original)")
            
            # Estimate token savings
            original_tokens = optimizer.get_token_estimate(resume_content + job_posting.description)
            optimized_tokens = optimizer.get_token_estimate(
                optimized_resume.optimized_text + optimized_job.optimized_text
            )
            token_savings = (original_tokens - optimized_tokens) / original_tokens
            
            print(f"🎯 Estimated Token Savings: {token_savings:.2%} "
                  f"({original_tokens:,} → {optimized_tokens:,} tokens)")
            
            results[config_name] = {
                'resume_compression': resume_compression,
                'job_compression': job_compression,
                'token_savings': token_savings,
                'original_tokens': original_tokens,
                'optimized_tokens': optimized_tokens
            }
        else:
            print("🚫 Token optimization disabled")
            results[config_name] = {
                'resume_compression': 1.0,
                'job_compression': 1.0,
                'token_savings': 0.0,
                'original_tokens': optimizer.get_token_estimate(resume_content + job_posting.description),
                'optimized_tokens': optimizer.get_token_estimate(resume_content + job_posting.description)
            }
        
        print()
    
    # Summary comparison
    print("=" * 70)
    print("  OPTIMIZATION COMPARISON SUMMARY")
    print("=" * 70)
    
    for config_name, result in results.items():
        print(f"{config_name}:")
        print(f"  Token Savings: {result['token_savings']:.2%}")
        print(f"  Resume Compression: {result['resume_compression']:.2%}")
        print(f"  Job Compression: {result['job_compression']:.2%}")
        print(f"  Total Tokens: {result['optimized_tokens']:,}")
        print()
    
    # Find best optimization
    best_config = max(results.keys(), key=lambda k: results[k]['token_savings'])
    print(f"🏆 Best Token Savings: {best_config} "
          f"({results[best_config]['token_savings']:.2%} reduction)")
    
    return results

def test_integration_with_ollama():
    """Test the integration with actual Ollama models (if available)."""
    print("=" * 70)
    print("  OLLAMA INTEGRATION TEST")
    print("=" * 70)
    
    job_posting = create_test_job_posting()
    resume_content = read_test_resume()
    
    if not resume_content:
        print("❌ Could not load resume template for testing")
        return
    
    # Test with different configurations
    configs = [
        ("Standard", SystemConfig.default()),
        ("Memory Optimized", SystemConfig.memory_optimized())
    ]
    
    for config_name, config in configs:
        print(f"Testing {config_name} with Ollama...")
        print("-" * 50)
        
        try:
            # Initialize Ollama integration
            ollama_client = OllamaIntegration(config=config)
            
            print(f"✅ Ollama connection established")
            print(f"🔧 Using model: {config.models.default_model}")
            print(f"⚙️  Token optimization: {'Enabled' if config.token_optimization.enabled else 'Disabled'}")
            
            # Test model availability
            available_models = ollama_client.list_available_models()
            required_models = [config.models.default_model, config.models.parsing_model]
            
            missing_models = [m for m in required_models if m not in available_models]
            if missing_models:
                print(f"⚠️  Missing models: {missing_models}")
                print("   Run: ollama pull <model_name> to install")
            else:
                print(f"✅ All required models available")
                
                # You could add actual generation tests here if models are available
                # For now, just validate the setup
                
        except Exception as e:
            print(f"❌ Ollama integration failed: {e}")
        
        print()

def main():
    """Run all optimization tests."""
    print("Starting Token Optimization Test Suite...")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run token optimization tests
    optimization_results = test_token_optimization()
    
    # Run integration tests
    test_integration_with_ollama()
    
    print("=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70)
    print("✅ Token optimization testing completed successfully!")
    
    # Provide recommendations
    if optimization_results:
        best_savings = max(r['token_savings'] for r in optimization_results.values())
        if best_savings > 0.3:  # 30% savings
            print("💡 Recommendation: Token optimization provides significant savings!")
            print("   Consider using Memory Optimized config for resource-constrained environments.")
        elif best_savings > 0.1:  # 10% savings
            print("💡 Recommendation: Token optimization provides moderate savings.")
            print("   Use Default config for balanced performance and quality.")
        else:
            print("💡 Recommendation: Token optimization provides minimal savings.")
            print("   Consider High Quality config for maximum output quality.")

if __name__ == "__main__":
    main()
