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

    def _get_fit_score_emoji_and_text(self, fit_score):
        """Get emoji and descriptive text based on fit score"""
        if fit_score >= 85:
            return "🟢", "Excellent Match"
        elif fit_score >= 75:
            return "🟡", "Strong Match"
        elif fit_score >= 60:
            return "🟠", "Moderate Match"
        else:
            return "🔴", "Weak Match"


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
                children=self._create_summary_blocks(job_data, summary_content)
            )
            
            print(f"✅ Job application created in Notion: {response['id']}")
            return response
            
        except Exception as e:
            print(f"❌ Error creating job application in Notion: {str(e)}")
            return None

    def update_job_application_status(self, page_id, new_status):
        """Update the status of an existing job application"""
        try:
            response = self.client.pages.update(
                page_id=page_id,
                properties={
                    "Status": {"status": {"name": new_status}}
                }
            )
            print(f"✅ Job application status updated to: {new_status}")
            return response
        except Exception as e:
            print(f"❌ Error updating job application status: {str(e)}")
            return None

    def get_job_applications(self, status_filter=None):
        """Retrieve job applications from the database, optionally filtered by status"""
        try:
            filter_condition = {}
            if status_filter:
                filter_condition = {
                    "property": "Status",
                    "status": {
                        "equals": status_filter
                    }
                }
            
            response = self.client.databases.query(
                database_id=NOTION_DATABASE_ID,
                filter=filter_condition if status_filter else None
            )
            
            print(f"✅ Retrieved {len(response['results'])} job applications")
            return response['results']
        except Exception as e:
            print(f"❌ Error retrieving job applications: {str(e)}")
            return []

    def _create_summary_blocks(self, job_data, summary_content):
        """Create structured Notion blocks for the job application summary"""
        blocks = []
        
        # Add main heading
        blocks.append({
            "object": "block",
            "type": "heading_1",
            "heading_1": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": f"📄 Job Application Summary"
                        },
                        "annotations": {
                            "bold": True
                        }
                    }
                ]
            }
        })
        
        # Add job details section
        blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": "🏢 Job Details"
                        },
                        "annotations": {
                            "bold": True
                        }
                    }
                ]
            }
        })
        
        # Add job information as bullet points
        fit_emoji, fit_text = self._get_fit_score_emoji_and_text(job_data["fit_score"])
        job_info_items = [
            f"**Position:** {job_data['name']}",
            f"**Company:** {job_data['company']}",
            f"**Application Date:** {job_data['start_date']}",
            f"**Fit Score:** {fit_emoji} {job_data['fit_score']}/100 ({fit_text})",
            f"**Status:** {job_data['status']}"
        ]
        
        for item in job_info_items:
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": item
                            }
                        }
                    ]
                }
            })
        
        # Add summary content if provided
        if summary_content and summary_content.strip():
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": "📋 Application Summary"
                            },
                            "annotations": {
                                "bold": True
                            }
                        }
                    ]
                }
            })
            
            # Split summary into paragraphs for better formatting
            summary_paragraphs = summary_content.split('\n\n')
            for paragraph in summary_paragraphs:
                if paragraph.strip():
                    blocks.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {
                                        "content": paragraph.strip()
                                    }
                                }
                            ]
                        }
                    })
        
        # Add additional sections if available in job_data
        if job_data.get("additional_context"):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": "ℹ️ Additional Context"
                            },
                            "annotations": {
                                "bold": True
                            }
                        }
                    ]
                }
            })
            
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": job_data["additional_context"]
                            }
                        }
                    ]
                }
            })
        
        # Add detailed fit assessment feedback if available
        if job_data.get("detailed_feedback"):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": "🎯 Detailed Fit Analysis"
                            },
                            "annotations": {
                                "bold": True
                            }
                        }
                    ]
                }
            })
            
            # Split detailed feedback into paragraphs
            feedback_paragraphs = job_data["detailed_feedback"].split('\n\n')
            for paragraph in feedback_paragraphs:
                if paragraph.strip():
                    blocks.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {
                                        "content": paragraph.strip()
                                    }
                                }
                            ]
                        }
                    })
        
        # Add score breakdown if available
        if job_data.get("score_breakdown"):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": "📊 Score Breakdown"
                            },
                            "annotations": {
                                "bold": True
                            }
                        }
                    ]
                }
            })
            
            for category, score in job_data["score_breakdown"].items():
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": f"**{category.replace('_', ' ').title()}:** {score}/100"
                                }
                            }
                        ]
                    }
                })
        
        if job_data.get("key_strengths"):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": "💪 Key Strengths"
                            },
                            "annotations": {
                                "bold": True
                            }
                        }
                    ]
                }
            })
            
            for strength in job_data["key_strengths"]:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": strength
                                }
                            }
                        ]
                    }
                })
        
        if job_data.get("recommendations"):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": "💡 Recommendations"
                            },
                            "annotations": {
                                "bold": True
                            }
                        }
                    ]
                }
            })
            
            for rec in job_data["recommendations"]:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": rec
                                }
                            }
                        ]
                    }
                })
        
        # Add missing skills section
        if job_data.get("missing_skills"):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": "📈 Areas for Improvement"
                            },
                            "annotations": {
                                "bold": True
                            }
                        }
                    ]
                }
            })
            
            for skill in job_data["missing_skills"]:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": skill
                                }
                            }
                        ]
                    }
                })
        
        # Add skills required section
        if job_data.get("skills_required"):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": "🛠️ Skills Required"
                            },
                            "annotations": {
                                "bold": True
                            }
                        }
                    ]
                }
            })
            
            # Create a paragraph with comma-separated skills
            skills_text = ", ".join(job_data["skills_required"])
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": skills_text
                            }
                        }
                    ]
                }
            })
        
        # Add location if provided
        if job_data.get("location"):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": "📍 Location"
                            },
                            "annotations": {
                                "bold": True
                            }
                        }
                    ]
                }
            })
            
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": job_data["location"]
                            }
                        }
                    ]
                }
            })
        
        # Add a divider and footer
        blocks.append({
            "object": "block",
            "type": "divider",
            "divider": {}
        })
        
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": f"🤖 Generated by Job Application Automator on {job_data['start_date']}"
                        },
                        "annotations": {
                            "italic": True,
                            "color": "gray"
                        }
                    }
                ]
            }
        })
        
        return blocks