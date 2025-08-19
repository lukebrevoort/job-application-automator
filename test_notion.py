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
    "summary": """This is a test job application for a Software Engineer position at Tech Company.

The role involves full-stack development using modern technologies including React, Node.js, and Python. The position offers excellent growth opportunities in a fast-paced startup environment.

Key requirements include 3+ years of experience in web development, strong problem-solving skills, and experience with cloud platforms like AWS or GCP.""",
    "key_strengths": [
        "Strong background in computer science with 3.96 GPA",
        "Experience with Python, JavaScript, and modern frameworks",
        "Proven leadership experience in student government",
        "Research experience in NLP and machine learning",
        "Full-stack development capabilities"
    ],
    "recommendations": [
        "Highlight machine learning research experience",
        "Emphasize full-stack development projects",
        "Mention leadership roles and responsibilities",
        "Showcase problem-solving abilities through projects"
    ],
    "location": "Remote",
    "skills_required": ["Python", "JavaScript", "React", "Node.js", "AWS", "Problem Solving"]
})
