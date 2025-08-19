#!/usr/bin/env python3
"""
Test script for the job scraper functionality.

This script tests the scraper's ability to extract information from
a simple HTML job posting without requiring external URLs.
"""

import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from job_scraper import JobScraper, JobPosting
import tempfile
import http.server
import socketserver
from threading import Thread
import time


def create_test_html():
    """Create a sample HTML job posting for testing."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Software Engineer - Test Company</title>
</head>
<body>
    <h1>Software Engineer</h1>
    <div class="company">Test Company Inc.</div>
    <div class="location">San Francisco, CA</div>
    <div class="job-description">
        <p>We are looking for a skilled Software Engineer to join our dynamic team.</p>
        
        <h3>Requirements:</h3>
        <ul>
            <li>Bachelor's degree in Computer Science or related field</li>
            <li>3+ years of experience with Python programming</li>
            <li>Experience with React and JavaScript frameworks</li>
            <li>Knowledge of AWS cloud services</li>
            <li>Strong problem-solving skills</li>
        </ul>
        
        <h3>Responsibilities:</h3>
        <ul>
            <li>Develop and maintain web applications</li>
            <li>Collaborate with cross-functional teams</li>
            <li>Write clean, maintainable code</li>
            <li>Participate in code reviews</li>
        </ul>
        
        <p>Join us and work with cutting-edge technologies including Docker, Kubernetes, 
        and machine learning frameworks like TensorFlow. Experience with SQL databases 
        and Git version control is essential.</p>
    </div>
</body>
</html>
    """


def start_test_server(html_content, port=8000):
    """Start a simple HTTP server serving the test HTML."""
    
    class TestHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html_content.encode())
    
    try:
        with socketserver.TCPServer(("", port), TestHTTPRequestHandler) as httpd:
            print(f"Test server running on http://localhost:{port}")
            httpd.serve_forever()
    except OSError:
        # Port might be in use, try the next one
        return start_test_server(html_content, port + 1)


def test_scraper_basic():
    """Test the scraper with a simple requests-based approach."""
    print("Testing basic scraping functionality...")
    
    # Test the skill extraction functionality
    test_text = """
    We are looking for a Python developer with experience in React, JavaScript, 
    AWS, Docker, and machine learning. Knowledge of SQL and Git is required.
    """
    
    scraper = JobScraper(use_selenium=False)
    skills = scraper._extract_skills_from_text(test_text)
    
    print(f"Extracted skills: {skills}")
    
    expected_skills = ['python', 'react', 'javascript', 'aws', 'docker', 'machine learning', 'sql', 'git']
    found_skills = [skill for skill in expected_skills if skill in skills]
    
    print(f"Found {len(found_skills)}/{len(expected_skills)} expected skills")
    
    # Test requirements extraction
    test_requirements = """
    Requirements:
    • Bachelor's degree in Computer Science
    • 3+ years of Python experience
    • Knowledge of web frameworks
    * Experience with cloud platforms
    1. Strong communication skills
    2. Problem-solving abilities
    """
    
    requirements = scraper._extract_requirements_from_text(test_requirements)
    print(f"Extracted requirements: {requirements}")
    
    print("✓ Basic scraping functionality test completed")


def test_scraper_with_server():
    """Test the scraper with a local test server."""
    print("\nTesting scraper with local server...")
    
    html_content = create_test_html()
    port = 8000
    
    # Start server in a separate thread
    server_thread = Thread(target=start_test_server, args=(html_content, port), daemon=True)
    server_thread.start()
    
    # Give server time to start
    time.sleep(2)
    
    test_url = f"http://localhost:{port}"
    
    try:
        # Test with requests first (faster)
        print("Testing with requests-based scraping...")
        with JobScraper(use_selenium=False) as scraper:
            job = scraper.scrape_job(test_url)
            
            print(f"Title: {job.title}")
            print(f"Company: {job.company}")
            print(f"Location: {job.location}")
            print(f"Skills found: {len(job.skills)}")
            print(f"Requirements found: {len(job.requirements)}")
            
            # Verify we got reasonable results
            if job.title and "Software Engineer" in job.title:
                print("✓ Title extraction successful")
            if job.company and "Test Company" in job.company:
                print("✓ Company extraction successful")
            if job.skills and len(job.skills) > 0:
                print("✓ Skills extraction successful")
            
    except Exception as e:
        print(f"✗ Error during server-based testing: {e}")


def test_selenium_setup():
    """Test if Selenium can be set up properly."""
    print("\nTesting Selenium setup...")
    
    try:
        scraper = JobScraper(use_selenium=True, headless=True)
        driver = scraper._setup_selenium()
        
        if driver:
            print("✓ Selenium Chrome driver setup successful")
            scraper.close()
        else:
            print("✗ Failed to setup Chrome driver")
            
    except Exception as e:
        print(f"✗ Selenium setup failed: {e}")
        print("Note: This might be expected if Chrome is not installed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Job Scraper Test Suite")
    print("=" * 60)
    
    # Test 1: Basic functionality
    test_scraper_basic()
    
    # Test 2: Server-based scraping  
    test_scraper_with_server()
    
    # Test 3: Selenium setup
    test_selenium_setup()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
