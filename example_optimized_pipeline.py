#!/usr/bin/env python3
"""
Token-Optimized Pipeline Example

This example shows how to use the new token optimization features
in your existing job application automation workflow.
"""

import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from job_scraper import JobScraper
from ollama_integration import OllamaIntegration
from config import SystemConfig

def main():
    """Demonstrate token-optimized job application automation."""
    print("=" * 70)
    print("  TOKEN-OPTIMIZED JOB APPLICATION AUTOMATION")
    print("=" * 70)
    
    # Test URL - replace with actual job posting URL
    test_url = "https://wd1.myworkdaysite.com/recruiting/paypal/jobs/job/San-Jose-California-United-States-of-America/Software-Engineer-Intern_R0129980"
    
    # Load resume template
    resume_path = os.path.join(os.path.dirname(__file__), 'templates', 'resume.tex')
    try:
        with open(resume_path, 'r', encoding='utf-8') as f:
            base_resume = f.read()
        print(f"📄 Loaded resume template ({len(base_resume):,} characters)")
    except FileNotFoundError:
        print("❌ Resume template not found. Please ensure templates/resume.tex exists.")
        return
    
    # Step 1: Compare different optimization configurations
    print("\n🔧 Testing different optimization configurations...")
    
    configs = {
        "No Optimization": OllamaIntegration(optimize_tokens=False),
        "Default Optimization": OllamaIntegration(config=SystemConfig.default()),
        "Memory Optimized": OllamaIntegration(config=SystemConfig.memory_optimized()),
        "High Quality": OllamaIntegration(config=SystemConfig.high_quality())
    }
    
    # Step 2: Scrape job posting
    print(f"\n🌐 Scraping job posting from: {test_url}")
    scraper = JobScraper()
    
    try:
        raw_content = scraper.scrape_job_posting(test_url)
        
        if not raw_content.success:
            print(f"❌ Failed to scrape job: {raw_content.error_message}")
            return
        
        print(f"✅ Scraped content successfully ({len(raw_content.cleaned_text):,} characters)")
        
        # Test each configuration
        for config_name, ollama_client in configs.items():
            print(f"\n--- Testing {config_name} ---")
            
            try:
                # Parse job posting
                print("📋 Parsing job details...")
                job_posting = ollama_client.parse_job_content(raw_content)
                
                print(f"  Title: {job_posting.title}")
                print(f"  Company: {job_posting.company}")
                print(f"  Skills: {len(job_posting.skills)} found")
                print(f"  Requirements: {len(job_posting.requirements)} found")
                
                # Personalize resume
                print("📝 Personalizing resume...")
                start_time = datetime.now()
                resume_result = ollama_client.personalize_resume(base_resume, job_posting)
                resume_time = (datetime.now() - start_time).total_seconds()
                
                # Calculate token savings
                if hasattr(ollama_client, 'token_optimizer') and ollama_client.token_optimizer:
                    original_tokens = ollama_client.token_optimizer.get_token_estimate(
                        base_resume + job_posting.description
                    )
                    optimized_tokens = ollama_client.token_optimizer.get_token_estimate(
                        resume_result.personalized_latex + job_posting.description
                    )
                    token_savings = (original_tokens - optimized_tokens) / original_tokens
                    
                    print(f"  📊 Token Usage: {optimized_tokens:,} (saved {token_savings:.1%})")
                else:
                    print(f"  📊 Token Usage: ~{len(base_resume + job_posting.description) // 4:,} (no optimization)")
                
                print(f"  ⏱️  Processing Time: {resume_time:.1f}s")
                print(f"  📄 Resume Length: {len(resume_result.personalized_latex):,} characters")
                
                # Generate cover letter
                print("💌 Generating cover letter...")
                start_time = datetime.now()
                cover_letter = ollama_client.generate_cover_letter(job_posting, base_resume)
                cover_letter_time = (datetime.now() - start_time).total_seconds()
                
                print(f"  ⏱️  Processing Time: {cover_letter_time:.1f}s")
                print(f"  📄 Cover Letter Length: {len(cover_letter.cover_letter_markdown):,} characters")
                
                # Assess fit score
                print("🎯 Assessing fit score...")
                start_time = datetime.now()
                fit_assessment = ollama_client.assess_fit_score(base_resume, job_posting)
                fit_time = (datetime.now() - start_time).total_seconds()
                
                print(f"  ⏱️  Processing Time: {fit_time:.1f}s")
                print(f"  🏆 Fit Score: {fit_assessment.fit_score}/100")
                
                total_time = resume_time + cover_letter_time + fit_time
                print(f"  📈 Total Time: {total_time:.1f}s")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to process job: {e}")
    
    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)
    print("💡 For regular use: Default Optimization (good balance)")
    print("💡 For large resumes or limited resources: Memory Optimized")
    print("💡 For maximum quality: High Quality")
    print("💡 For comparison/debugging: No Optimization")
    
    print("\n✅ Token optimization testing complete!")
    print("   See TOKEN_OPTIMIZATION_README.md for more details.")

if __name__ == "__main__":
    main()
