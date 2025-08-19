"""
Job Posting Scraper

This module provides a simplified web scraper that extracts raw HTML content
from job posting sites. The actual parsing of job details (title, company, 
skills, requirements) is handled by LLM in the Ollama integration module.

This approach is more robust and flexible than trying to parse with CSS selectors.
"""

import time
import logging
import os
from typing import Optional
from urllib.parse import urlparse
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RawJobContent:
    """Data structure to hold raw scraped content before LLM parsing."""
    url: str
    raw_html: str = ""
    cleaned_text: str = ""
    page_title: str = ""
    domain: str = ""
    success: bool = True
    error_message: str = ""


@dataclass 
class JobPosting:
    """Data structure to hold parsed job information (populated by LLM)."""
    url: str
    title: str = ""
    company: str = ""
    location: str = ""
    description: str = ""
    requirements: list = None
    skills: list = None
    salary_range: str = ""
    job_type: str = ""
    experience_level: str = ""
    posted_date: str = ""
    raw_content: RawJobContent = None
    
    def __post_init__(self):
        if self.requirements is None:
            self.requirements = []
        if self.skills is None:
            self.skills = []


class JobScraper:
    """
    A simplified job scraper that extracts raw HTML content for LLM processing.
    
    Instead of trying to parse job details with CSS selectors, this scraper
    focuses on getting clean, readable content that can be processed by an LLM.
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
        
        # Try to find Chrome binary in common locations
        chrome_paths = [
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            '/Applications/Chromium.app/Contents/MacOS/Chromium',
            '/usr/bin/google-chrome',
            '/usr/bin/chromium-browser',
        ]
        
        for chrome_path in chrome_paths:
            if os.path.exists(chrome_path):
                chrome_options.binary_location = chrome_path
                logger.info(f"Found Chrome at: {chrome_path}")
                break
        else:
            logger.warning("Chrome binary not found in common locations")
        
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
        try:
            driver = self._setup_selenium()
        except Exception as e:
            logger.error(f"Failed to setup Selenium: {e}")
            logger.info("Falling back to requests-based scraping")
            return self._get_content_requests(url)
        
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
            logger.info("Falling back to requests-based scraping")
            return self._get_content_requests(url)
    
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
    
    def _clean_html_content(self, soup: BeautifulSoup) -> str:
        """
        Extract clean, readable text from HTML content.
        Removes navigation, ads, and other non-essential elements.
        """
        # Remove script and style elements
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        
        # Remove common navigation and ad elements
        unwanted_selectors = [
            'nav', 'header', 'footer', '.nav', '.navigation',
            '.ad', '.ads', '.advertisement', '[class*="ad-"]',
            '.cookie', '.gdpr', '.popup', '.modal', '.overlay',
            '.sidebar', '.widget', '.share', '.social',
            '.breadcrumb', '.pagination', '.comments'
        ]
        
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Get text content
        text = soup.get_text(separator='\\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.split('\\n') if line.strip()]
        cleaned_text = '\\n'.join(lines)
        
        return cleaned_text
    
    def scrape_raw_content(self, url: str) -> RawJobContent:
        """
        Extract raw content from a job posting URL.
        
        Args:
            url: The URL of the job posting to scrape
            
        Returns:
            RawJobContent object containing raw HTML and cleaned text
        """
        logger.info(f"Scraping raw content from: {url}")
        
        try:
            # Get page content - try Selenium first, fall back to requests
            try:
                soup = self._get_page_content(url)
            except Exception as e:
                logger.warning(f"Primary scraping method failed: {e}")
                logger.info("Trying requests-only approach...")
                soup = self._get_content_requests(url)
            
            domain = self._get_domain(url)
            
            # Extract page title
            page_title = ""
            title_tag = soup.find('title')
            if title_tag:
                page_title = title_tag.get_text(strip=True)
            
            # Clean the HTML content
            cleaned_text = self._clean_html_content(soup)
            
            # Store raw HTML (limited to prevent memory issues)
            raw_html = str(soup)[:10000]  # First 10k chars
            
            # Validate that we got meaningful content
            if len(cleaned_text.strip()) < 50:
                logger.warning("Very little content extracted, might be blocked or invalid page")
            
            return RawJobContent(
                url=url,
                raw_html=raw_html,
                cleaned_text=cleaned_text,
                page_title=page_title,
                domain=domain,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to scrape content from {url}: {e}")
            return RawJobContent(
                url=url,
                domain=self._get_domain(url),
                success=False,
                error_message=str(e)
            )
    
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
    """Example usage of the simplified job scraper."""
    # Example URLs for testing
    test_urls = [
        "https://www.linkedin.com/jobs/view/4258759674/?alternateChannel=search&eBP=NOT_ELIGIBLE_FOR_CHARGING&refId=eFErWbVbRbeWPjGYZfcj6A%3D%3D&trackingId=rnjMW7RFFZGVmmGkjI2%2BMQ%3D%3D",
        "https://careers.datadoghq.com/detail/7127832/?gh_jid=7127832",
        "https://jobs.careers.microsoft.com/global/en/job/1861019/Software-Engineer:-Frontend-Intern-Opportunities-for-University-Students,-Redmond?amp;utm_campaign=Copy-job-share",
    ]
    
    with JobScraper(use_selenium=True, headless=True) as scraper:
        for url in test_urls:
            try:
                raw_content = scraper.scrape_raw_content(url)
                print(f"\\nURL: {url}")
                print(f"Domain: {raw_content.domain}")
                print(f"Page Title: {raw_content.page_title}")
                print(f"Success: {raw_content.success}")
                print(f"Content Length: {len(raw_content.cleaned_text)} characters")
                print(f"Sample Content: {raw_content.cleaned_text[:200]}...")
                print("-" * 70)
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")


if __name__ == "__main__":
    main()
