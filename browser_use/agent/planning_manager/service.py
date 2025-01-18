from __future__ import annotations

import logging
from typing import Optional, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from browser_use.agent.planning_manager.views import TaskPlan

logger = logging.getLogger(__name__)

class PlanningManager:
    def __init__(
        self,
        llm: BaseChatModel,
    ):
        self.llm = llm

    def _get_planning_system_prompt(self) -> SystemMessage:
        """Get system prompt for task planning"""
        return SystemMessage(content="""You are a task planning assistant. Your role is to:
1. Analyze the given task description
2. Create a well-structured, clear version of the task
3. Add relevant metadata like tags and difficulty estimation

Respond with a structured task plan that includes:
- A reformulated task description that is clear and actionable
- A short summary of the task
- Relevant tags/categories
- Estimated difficulty (1-10)

Keep the reformulated task focused and specific while maintaining all important details from the original task.""")

    async def plan_task(self, raw_task: str) -> TaskPlan:
        """Generate structured task plan from raw task description"""
        
        # Create messages for the LLM
        messages = [
            self._get_planning_system_prompt(),
            HumanMessage(content=f"Please analyze and structure this task: {raw_task}")
        ]

        # Get structured output using TaskPlan model
        structured_llm = self.llm.with_structured_output(TaskPlan, include_raw=True)
        response: dict[str, Any] = await structured_llm.ainvoke(messages)  # type: ignore
        task_plan: Optional[TaskPlan] = response['parsed']

        if task_plan is None:
            raise ValueError("Could not parse task plan response")

        logger.info(f"Generated task plan with difficulty {task_plan.estimated_difficulty}")
        logger.debug(f"Task tags: {', '.join(task_plan.tags)}")
        
        return task_plan
    