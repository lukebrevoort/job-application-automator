# Job Application Automator

An intelligent job application automation tool that leverages AI to streamline the job search process. This project automatically scrapes job postings, generates personalized resumes and cover letters, assesses job fit scores, and manages applications through Notion integration.

## Features

- **Job Scraping**: Extracts job posting details from various job boards using Selenium and BeautifulSoup
- **AI-Powered Personalization**: Uses Ollama's GPT-OSS 20B model to:
  - Personalize LaTeX resumes based on job requirements
  - Generate custom cover letters using your template
  - Calculate job fit scores with detailed analysis
- **LaTeX Resume Processing**: Automatically modifies your LaTeX resume to highlight relevant skills and experience
- **Notion Integration**: Tracks applications, fit scores, and status in a structured Notion database
- **Professional Output**: Generates clean, professional documents ready for submission

## Project Structure

```
├── src/                           # Main source code
│   ├── job_scraper.py            # Web scraping functionality
│   ├── ollama_integration.py     # AI/LLM integration
│   ├── latex_resume_processor.py # LaTeX resume personalization
│   ├── cover_letter_generator.py # Cover letter generation
│   └── notion.py                 # Notion API integration
├── templates/                     # Document templates
│   ├── coverLetter.md            # Cover letter template
│   └── resume.tex                # LaTeX resume template
├── config/                        # Configuration files
├── output/                        # Generated documents
└── test_*.py                     # Integration tests
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- Chrome browser (for web scraping)
- LaTeX distribution (for resume processing)
- Notion account (optional, for application tracking)

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd job-application-automator
pip install -r requirements.txt
```

### 2. Install and Configure Ollama

```bash
# Install Ollama (macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull gpt-oss:20b
ollama pull llama3.2:3b
```

### 3. Set Up LaTeX (for resume processing)

**macOS:**
```bash
brew install --cask mactex
```

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-full
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# Notion Integration (optional)
NOTION_SECRET=your_notion_integration_token
NOTION_DATABASE_ID=your_database_id

# Additional configurations as needed
```

### 5. Prepare Your Templates

1. **Resume Template**: Place your LaTeX resume template in `templates/resume.tex`
2. **Cover Letter Template**: Customize `templates/coverLetter.md` with your information

### 6. Install Chrome WebDriver

The project uses webdriver-manager to automatically handle Chrome driver installation, but ensure Chrome browser is installed on your system.

## Usage

### Basic Pipeline Test

Run the full pipeline test to see the system in action:

```bash
python test_full_pipeline.py
```

This will:
1. Scrape a sample job posting
2. Generate a personalized resume
3. Create a custom cover letter
4. Calculate a fit score
5. Save all outputs to the `output/` directory

### Individual Components

You can also test individual components:

```bash
# Test job scraping
python test_scraper.py

# Test Notion integration
python test_notion.py

# Test enhanced fit assessment
python test_enhanced_fit_assessment.py
```

### Custom Job Application

```python
from src.job_scraper import JobScraper
from src.ollama_integration import OllamaIntegration
from src.cover_letter_generator import CoverLetterGenerator

# Scrape a job posting
scraper = JobScraper()
job = scraper.scrape_job("https://example-job-posting-url.com")

# Generate personalized documents
ollama = OllamaIntegration()
cover_letter_gen = CoverLetterGenerator()

# Create personalized content
fit_score = ollama.assess_job_fit(job, "path/to/your/resume.tex")
cover_letter = cover_letter_gen.generate_cover_letter(job)
```

## Configuration

### Notion Database Setup (Optional)

If you want to track applications in Notion:

1. Create a new Notion database
2. Add properties for: Name, Company, Fit Score, Status, Start Date, URL
3. Create a new integration in Notion settings
4. Add the integration token and database ID to your `.env` file

### Model Configuration

The system uses different Ollama models for different tasks:
- `gpt-oss:20b`: General writing and analysis tasks
- `llama3.2:3b`: Fast parsing tasks
- `codellama`: Technical content analysis

You can modify model selection in `src/ollama_integration.py`.

## Output

Generated files are saved to timestamped directories in `output/`:
- `personalized_resume.tex`: LaTeX resume tailored to the job
- `personalized_cover_letter.md`: Custom cover letter
- `application_summary.md`: Job analysis and fit assessment
- `personalized_resume_metadata.json`: Processing metadata

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**: Ensure Ollama is running (`ollama serve`)
2. **LaTeX Compilation Issues**: Install a complete LaTeX distribution
3. **Chrome Driver Issues**: Ensure Chrome browser is installed and up to date
4. **Notion API Errors**: Verify your integration token and database permissions

### Dependencies

If you encounter issues with PyLaTeX, the requirements.txt uses the git version for Python 3.12 compatibility:

```bash
pip install git+https://github.com/JelteF/PyLaTeX.git
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests with `pytest`
4. Submit a pull request

## License

This project is open source. Please check the license file for details.