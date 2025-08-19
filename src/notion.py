"""
This file is responsible for building the functions which will interact with the 
Notion API for managing job application data.
This includes functions for creating, updating, and retrieving job application records,
as well as managing related tasks and notes.
"""

import os
from notion_client import Client
from dotenv import load_dotenv

load_dotenv()

NOTION_SECRET = os.getenv("NOTION_SECRET")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

class Notion:

    def __init__(self):
        if not NOTION_SECRET:
            raise ValueError("NOTION_SECRET environment variable is not set")
        if not NOTION_DATABASE_ID:
            raise ValueError("NOTION_DATABASE_ID environment variable is not set")
            
        self.client = Client(auth=NOTION_SECRET)

    def validate_job_data(self, job_data):
        """Validate required fields in job_data"""
        required_fields = ["name", "company", "fit_score", "status", "start_date", "url"]
        for field in required_fields:
            if field not in job_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate data types
        if not isinstance(job_data["fit_score"], (int, float)):
            raise ValueError("fit_score must be a number")
        
        if job_data["fit_score"] < 0 or job_data["fit_score"] > 100:
            raise ValueError("fit_score must be between 0 and 100")


    def create_job_application(self, job_data):
        try:
            # Validate input data
            self.validate_job_data(job_data)
            
            # Convert complex objects to strings if needed
            resume_content = str(job_data.get("resume", "")) if not isinstance(job_data.get("resume", ""), str) else job_data.get("resume", "")
            cover_letter_content = str(job_data.get("cover_letter", "")) if not isinstance(job_data.get("cover_letter", ""), str) else job_data.get("cover_letter", "")
            summary_content = str(job_data.get("summary", "")) if not isinstance(job_data.get("summary", ""), str) else job_data.get("summary", "")
            
            # Truncate long content for Notion rich text limits (2000 chars)
            resume_content = resume_content[:2000] if len(resume_content) > 2000 else resume_content
            cover_letter_content = cover_letter_content[:2000] if len(cover_letter_content) > 2000 else cover_letter_content
            summary_content = summary_content[:2000] if len(summary_content) > 2000 else summary_content
            
            response = self.client.pages.create(
                parent={"database_id": NOTION_DATABASE_ID},
                properties={
                    "Title": {"title": [{"text": {"content": job_data["name"]}}]},
                    "Company": {"rich_text": [{"text": {"content": job_data["company"]}}]},
                    "Fit Score": {"number": job_data["fit_score"]},
                    "Status": {"status": {"name": job_data["status"]}},
                    "Start Date": {"date": {"start": job_data["start_date"]}},
                    "Cover Letter": {"rich_text": [{"text": {"content": cover_letter_content}}]},
                    "URL": {"url": job_data["url"]},
                },
                children=[
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {
                                        "content": summary_content
                                    }
                                }
                            ]
                        }
                    }
                ] if summary_content else []
            )
            
            print(f"✅ Job application created in Notion: {response['id']}")
            return response
            
        except Exception as e:
            print(f"❌ Error creating job application in Notion: {str(e)}")
            return None