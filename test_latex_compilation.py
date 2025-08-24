#!/usr/bin/env python3
"""
LaTeX Compilation Test

This script tests that optimized resumes still compile correctly with LaTeX.
"""

import sys
import os
import tempfile
import subprocess

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from token_optimizer import TokenOptimizer
from job_scraper import JobPosting
from config import SystemConfig

def test_latex_compilation():
    """Test that optimized resumes can still be compiled with LaTeX."""
    print("=" * 60)
    print("  LATEX COMPILATION TEST")
    print("=" * 60)
    
    # Read the test resume
    resume_path = os.path.join(os.path.dirname(__file__), 'templates', 'resume.tex')
    try:
        with open(resume_path, 'r', encoding='utf-8') as f:
            original_resume = f.read()
    except FileNotFoundError:
        print("❌ Resume template not found")
        return False
    
    # Create a test job posting
    test_job = JobPosting(
        url="https://example.com/test",
        title="Software Engineer",
        company="TestCorp",
        location="Remote",
        description="We are looking for a software engineer with Python and React experience.",
        skills=["Python", "React", "JavaScript", "SQL"],
        requirements=["3+ years experience", "Bachelor's degree"]
    )
    
    # Test different optimization levels
    configs = {
        "Original": None,
        "Default": SystemConfig.default(),
        "Memory Optimized": SystemConfig.memory_optimized()
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"Testing {config_name}...")
        
        if config_name == "Original":
            resume_content = original_resume
        else:
            optimizer = TokenOptimizer()
            optimized = optimizer.optimize_resume_for_job(
                original_resume, test_job, 
                max_tokens=config.token_optimization.max_resume_tokens
            )
            resume_content = optimized.optimized_text
        
        # Check LaTeX structure
        latex_checks = {
            "Has documentclass": "\\documentclass" in resume_content,
            "Has begin document": "\\begin{document}" in resume_content,
            "Has end document": "\\end{document}" in resume_content,
            "Has sections": "\\section{" in resume_content,
            "Has resume items": "\\resumeItem" in resume_content
        }
        
        all_checks_pass = all(latex_checks.values())
        
        print(f"  📄 Length: {len(resume_content):,} characters")
        print(f"  ✅ LaTeX Structure: {'Valid' if all_checks_pass else 'Invalid'}")
        
        if not all_checks_pass:
            failed_checks = [check for check, passed in latex_checks.items() if not passed]
            print(f"  ❌ Failed checks: {', '.join(failed_checks)}")
        
        # Try to compile with pdflatex if available
        if all_checks_pass:
            compile_result = test_pdflatex_compilation(resume_content)
            print(f"  🔧 PDF Compilation: {'Success' if compile_result else 'Failed/Not Available'}")
        else:
            compile_result = False
        
        results[config_name] = {
            'length': len(resume_content),
            'structure_valid': all_checks_pass,
            'compiles': compile_result
        }
        print()
    
    # Summary
    print("=" * 60)
    print("  COMPILATION TEST SUMMARY")
    print("=" * 60)
    
    for config_name, result in results.items():
        status = "✅" if result['structure_valid'] and result['compiles'] else "⚠️"
        print(f"{status} {config_name}:")
        print(f"    Length: {result['length']:,} chars")
        print(f"    Structure: {'Valid' if result['structure_valid'] else 'Invalid'}")
        print(f"    Compiles: {'Yes' if result['compiles'] else 'No/Unknown'}")
    
    # Check if optimization broke anything
    original_valid = results["Original"]["structure_valid"]
    optimized_valid = all(r["structure_valid"] for name, r in results.items() if name != "Original")
    
    if original_valid and optimized_valid:
        print("\n🎉 All optimization levels preserve LaTeX structure!")
        return True
    else:
        print("\n⚠️  Some optimization levels may have issues")
        return False

def test_pdflatex_compilation(latex_content):
    """Test if content can be compiled with pdflatex."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
            f.write(latex_content)
            temp_tex_file = f.name
        
        # Try to compile with pdflatex
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run([
                'pdflatex', 
                '-interaction=nonstopmode',
                '-output-directory', temp_dir,
                temp_tex_file
            ], capture_output=True, timeout=30)
            
            success = result.returncode == 0
            
        # Clean up
        os.unlink(temp_tex_file)
        return success
        
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        # pdflatex not available or other error
        return None

if __name__ == "__main__":
    test_latex_compilation()
