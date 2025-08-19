"""
Job Posting Scraper

This module provides a flexible web scraper that can extract job information
from various job posting sites including LinkedIn, Indeed, company websites,
and other common job boards.

The scraper uses multiple strategies:
1. Site-specific selectors for known job sites
2. Semantic HTML parsing for generic sites
3. Fallback content extraction using AI-friendly text processing
"""

import re
import time
import json
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup, Tag
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class JobPosting:
    """Data structure to hold extracted job information."""
    url: str
    title: str = ""
    company: str = ""
    location: str = ""
    description: str = ""
    requirements: List[str] = None
    skills: List[str] = None
    salary_range: str = ""
    job_type: str = ""  # Full-time, Part-time, Contract, etc.
    experience_level: str = ""  # Entry, Mid, Senior, etc.
    posted_date: str = ""
    raw_html: str = ""
    
    def __post_init__(self):
        if self.requirements is None:
            self.requirements = []
        if self.skills is None:
            self.skills = []


class JobScraper:
    """
    A flexible job scraper that can handle multiple job posting sites.
    
    Uses both static scraping (requests + BeautifulSoup) and dynamic scraping 
    (Selenium) depending on the site's requirements.
    """
    
    def __init__(self, use_selenium: bool = True, headless: bool = True):
        self.use_selenium = use_selenium
        self.headless = headless
        self.session = requests.Session()
        self.driver = None
        
        # Set up headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Site-specific selectors for known job boards
        self.site_selectors = {
            'linkedin.com': {
                'title': ['.top-card-layout__title', '.t-24', 'h1'],
                'company': ['.topcard__org-name-link', '.topcard__flavor--bullet', '.sub-nav-cta__optional-url'],
                'location': ['.topcard__flavor--bullet', '.job-search-card__location'],
                'description': ['.description__text', '.show-more-less-html__markup'],
                'requirements': ['.description__text ul li', '.show-more-less-html__markup ul li'],
            },
            'indeed.com': {
                'title': ['[data-testid="jobsearch-JobInfoHeader-title"]', '.jobsearch-JobInfoHeader-title', 'h1'],
                'company': ['[data-testid="inlineHeader-companyName"]', '.jobsearch-InlineCompanyRating', 'span[title]'],
                'location': ['[data-testid="job-location"]', '.jobsearch-JobInfoHeader-subtitle'],
                'description': ['#jobDescriptionText', '.jobsearch-jobDescriptionText'],
                'requirements': ['#jobDescriptionText ul li', '.jobsearch-jobDescriptionText ul li'],
            },
            'glassdoor.com': {
                'title': ['.job-search-key-cfuokn', 'h1', '[data-test="job-title"]'],
                'company': ['[data-test="employer-name"]', '.job-search-key-cfuokn'],
                'location': ['[data-test="job-location"]', '.job-search-key-cfuokn'],
                'description': ['[data-test="jobDescriptionContainer"]', '.job-description-content'],
                'requirements': ['[data-test="jobDescriptionContainer"] ul li', '.job-description-content ul li'],
            },
            'lever.co': {
                'title': ['.posting-headline h2', 'h2'],
                'company': ['.posting-headline .company-name', '.company-name'],
                'location': ['.posting-categories .location', '.location'],
                'description': ['.posting-content .section-wrapper', '.posting-content'],
                'requirements': ['.posting-content ul li', '.section-wrapper ul li'],
            },
            'greenhouse.io': {
                'title': ['#header h1', 'h1'],
                'company': ['#header .company-name', '.company-name'],
                'location': ['#header .location', '.location'],
                'description': ['#content', '.application-detail'],
                'requirements': ['#content ul li', '.application-detail ul li'],
            },
        }
        
        # Common skill keywords to extract
        self.skill_keywords = [
            # Programming languages
            'python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'php', 'go', 'rust',
            'typescript', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql',
            
            # Frameworks and libraries
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
            'spring', 'rails', 'laravel', 'tensorflow', 'pytorch', 'pandas',
            
            # Tools and technologies  
            'git', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'jenkins',
            'terraform', 'ansible', 'mongodb', 'postgresql', 'mysql', 'redis',
            
            # Concepts and methodologies
            'machine learning', 'artificial intelligence', 'data science',
            'agile', 'scrum', 'devops', 'ci/cd', 'microservices', 'api',
        ]
    
    def _setup_selenium(self) -> webdriver.Chrome:
        """Set up Selenium Chrome driver with appropriate options."""
        if self.driver:
            return self.driver
            
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            return self.driver
        except Exception as e:
            logger.error(f"Failed to setup Selenium: {e}")
            raise
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).netloc.lower()
    
    def _get_page_content(self, url: str, use_selenium: bool = None) -> BeautifulSoup:
        """
        Get page content using either requests or Selenium.
        
        Args:
            url: The URL to scrape
            use_selenium: Override the default selenium usage for this request
        
        Returns:
            BeautifulSoup object containing the page content
        """
        should_use_selenium = use_selenium if use_selenium is not None else self.use_selenium
        domain = self._get_domain(url)
        
        # Some sites require Selenium due to heavy JavaScript usage
        js_heavy_sites = ['linkedin.com', 'glassdoor.com', 'indeed.com']
        if any(site in domain for site in js_heavy_sites):
            should_use_selenium = True
        
        if should_use_selenium:
            return self._get_content_selenium(url)
        else:
            return self._get_content_requests(url)
    
    def _get_content_requests(self, url: str) -> BeautifulSoup:
        """Get page content using requests."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Failed to fetch content with requests: {e}")
            raise
    
    def _get_content_selenium(self, url: str) -> BeautifulSoup:
        """Get page content using Selenium."""
        driver = self._setup_selenium()
        
        try:
            logger.info(f"Loading page with Selenium: {url}")
            driver.get(url)
            
            # Wait for page to load
            time.sleep(3)
            
            # Try to wait for common content indicators
            try:
                WebDriverWait(driver, 10).until(
                    lambda d: d.find_element(By.TAG_NAME, "body")
                )
            except TimeoutException:
                logger.warning("Timeout waiting for page to load")
            
            # Handle common pop-ups and overlays
            self._handle_popups(driver)
            
            page_source = driver.page_source
            return BeautifulSoup(page_source, 'html.parser')
            
        except Exception as e:
            logger.error(f"Failed to fetch content with Selenium: {e}")
            raise
    
    def _handle_popups(self, driver: webdriver.Chrome):
        """Handle common popups and overlays."""
        popup_selectors = [
            # LinkedIn
            '.authentication-outlet',
            '.guest-nav-modal',
            # Indeed
            '.popover-x-button-close',
            # General
            '[aria-label="Close"]',
            '.close-button',
            '.modal-close',
        ]
        
        for selector in popup_selectors:
            try:
                element = driver.find_element(By.CSS_SELECTOR, selector)
                if element.is_displayed():
                    element.click()
                    time.sleep(1)
            except:
                continue
    
    def _extract_with_selectors(self, soup: BeautifulSoup, selectors: List[str]) -> str:
        """Extract text using multiple CSS selectors as fallbacks."""
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text(strip=True)
                    if text:
                        return text
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
                continue
        return ""
    
    def _extract_list_with_selectors(self, soup: BeautifulSoup, selectors: List[str]) -> List[str]:
        """Extract list of text items using multiple CSS selectors."""
        items = []
        for selector in selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    if text and text not in items:
                        items.append(text)
            except Exception as e:
                logger.debug(f"List selector {selector} failed: {e}")
                continue
        return items
    
    def _extract_generic_job_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract job information using generic semantic HTML parsing.
        This is used as a fallback when site-specific selectors aren't available.
        """
        result = {}
        
        # Try to find title in common locations
        title_selectors = ['h1', '.job-title', '[class*="title"]', '[id*="title"]']
        result['title'] = self._extract_with_selectors(soup, title_selectors)
        
        # Try to find company name
        company_selectors = [
            '.company', '.employer', '[class*="company"]', '[class*="employer"]',
            '[aria-label*="company"]', 'span[title]'
        ]
        result['company'] = self._extract_with_selectors(soup, company_selectors)
        
        # Try to find location
        location_selectors = [
            '.location', '[class*="location"]', '[aria-label*="location"]'
        ]
        result['location'] = self._extract_with_selectors(soup, location_selectors)
        
        # Extract main content - look for largest text blocks
        content_selectors = [
            '.job-description', '.description', '[class*="description"]',
            '.content', '.job-content', 'main', '.main-content'
        ]
        
        description = ""
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                desc_text = element.get_text(separator=' ', strip=True)
                if len(desc_text) > len(description):
                    description = desc_text
        
        result['description'] = description
        
        return result
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract technical skills from job description text."""
        if not text:
            return []
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.skill_keywords:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        
        return found_skills
    
    def _extract_requirements_from_text(self, text: str) -> List[str]:
        """Extract requirements from job description text."""
        if not text:
            return []
        
        requirements = []
        
        # Look for bullet points and numbered lists
        bullet_patterns = [
            r'[•·▪▫‣⁃](.+?)(?=\n|[•·▪▫‣⁃]|$)',
            r'\*(.+?)(?=\n|\*|$)',
            r'\d+\.(.+?)(?=\n|\d+\.|$)',
        ]
        
        for pattern in bullet_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                req = match.strip()
                if len(req) > 10 and len(req) < 200:  # Filter out very short/long items
                    requirements.append(req)
        
        # Look for common requirement patterns
        requirement_patterns = [
            r'(?:requirement|required|must have|essential)[:.]?\s*(.+?)(?=\n|\.|;)',
            r'(?:experience|knowledge) (?:in|with|of)\s+(.+?)(?=\n|\.|;)',
            r'(?:bachelor|master|degree) (?:in|of)\s+(.+?)(?=\n|\.|;)',
        ]
        
        for pattern in requirement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                req = match.strip()
                if len(req) > 5 and len(req) < 100:
                    requirements.append(req)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_requirements = []
        for req in requirements:
            if req.lower() not in seen:
                seen.add(req.lower())
                unique_requirements.append(req)
        
        return unique_requirements[:10]  # Limit to top 10 requirements
    
    def scrape_job(self, url: str) -> JobPosting:
        """
        Main method to scrape a job posting from any supported URL.
        
        Args:
            url: The URL of the job posting to scrape
            
        Returns:
            JobPosting object containing extracted information
        """
        logger.info(f"Scraping job posting: {url}")
        
        try:
            # Get page content
            soup = self._get_page_content(url)
            domain = self._get_domain(url)
            
            # Initialize job posting
            job = JobPosting(url=url, raw_html=str(soup)[:5000])  # Store first 5k chars
            
            # Try site-specific extraction first
            extracted = False
            for site_domain, selectors in self.site_selectors.items():
                if site_domain in domain:
                    logger.info(f"Using site-specific selectors for {site_domain}")
                    
                    job.title = self._extract_with_selectors(soup, selectors.get('title', []))
                    job.company = self._extract_with_selectors(soup, selectors.get('company', []))
                    job.location = self._extract_with_selectors(soup, selectors.get('location', []))
                    job.description = self._extract_with_selectors(soup, selectors.get('description', []))
                    job.requirements = self._extract_list_with_selectors(soup, selectors.get('requirements', []))
                    
                    extracted = True
                    break
            
            # Fall back to generic extraction
            if not extracted or not job.title:
                logger.info("Using generic extraction methods")
                generic_data = self._extract_generic_job_info(soup)
                
                if not job.title:
                    job.title = generic_data.get('title', '')
                if not job.company:
                    job.company = generic_data.get('company', '')
                if not job.location:
                    job.location = generic_data.get('location', '')
                if not job.description:
                    job.description = generic_data.get('description', '')
            
            # Extract skills from description
            job.skills = self._extract_skills_from_text(job.description)
            
            # Extract requirements if not already found
            if not job.requirements:
                job.requirements = self._extract_requirements_from_text(job.description)
            
            # Clean up extracted data
            job.title = job.title.strip()
            job.company = job.company.strip()
            job.location = job.location.strip()
            job.description = ' '.join(job.description.split())  # Normalize whitespace
            
            logger.info(f"Successfully extracted job: {job.title} at {job.company}")
            return job
            
        except Exception as e:
            logger.error(f"Failed to scrape job posting {url}: {e}")
            # Return minimal job posting with error info
            return JobPosting(url=url, title=f"Error: {str(e)}")
    
    def close(self):
        """Clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Example usage of the job scraper."""
    # Example URLs for testing
    test_urls = [
        "https://www.linkedin.com/jobs/view/4258759674/?alternateChannel=search&eBP=NOT_ELIGIBLE_FOR_CHARGING&refId=eFErWbVbRbeWPjGYZfcj6A%3D%3D&trackingId=rnjMW7RFFZGVmmGkjI2%2BMQ%3D%3D",  # Replace with actual URLs
        "https://careers.datadoghq.com/detail/7127832/?gh_jid=7127832",
        "https://jobs.careers.microsoft.com/global/en/job/1861019/Software-Engineer:-Frontend-Intern-Opportunities-for-University-Students,-Redmond?amp;utm_campaign=Copy-job-share",
    ]
    
    with JobScraper(use_selenium=True, headless=True) as scraper:
        for url in test_urls:
            try:
                job = scraper.scrape_job(url)
                print(f"\nTitle: {job.title}")
                print(f"Company: {job.company}")
                print(f"Location: {job.location}")
                print(f"Skills: {', '.join(job.skills[:5])}")
                print(f"Requirements: {len(job.requirements)} found")
                print("-" * 50)
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")


if __name__ == "__main__":
    main()
