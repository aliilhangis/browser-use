from pydantic import BaseModel, Field

class TaskPlan(BaseModel):
    """Structured task plan with rich metadata"""
    task: str = Field(..., description="Reformulated version of the user's original task")
    short_summary: str = Field(..., description="A concise description of what the task entails")
    tags: list[str] = Field(..., description="Relevant keywords or categories associated with the task")
    estimated_difficulty: int = Field(
        ..., 
        description="Rough estimate of complexity level (1-10)",
     
    )
