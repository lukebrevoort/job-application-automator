from src.notion import Notion

def read_file_content(file_path):
    """Read and return the content of a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

notion_client = Notion()

# Read LaTeX and Markdown content
resume_content = read_file_content("/Users/lbrevoort/Desktop/projects/job-application-automator/templates/resume.tex")
cover_letter_content = read_file_content("/Users/lbrevoort/Desktop/projects/job-application-automator/templates/coverLetter.md")

notion_client.create_job_application({
    "name": "Software Engineer",
    "company": "Tech Company",
    "fit_score": 85,
    "status": "Applied",
    "start_date": "2025-08-19",
    "resume": resume_content,
    "cover_letter": cover_letter_content,
    "url": "https://example.com/job-posting",
    "summary": "Brief summary of the job application"
})
