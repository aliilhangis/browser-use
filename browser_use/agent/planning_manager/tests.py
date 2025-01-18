from langchain_anthropic import ChatAnthropic
import pytest
from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from browser_use.agent.planning_manager.service import PlanningManager
from browser_use.agent.planning_manager.views import TaskPlan


@pytest.fixture(
	params=[
		ChatOpenAI(model='gpt-4o-mini'),
		AzureChatOpenAI(model='gpt-4o', api_version='2024-02-15-preview'),
		ChatAnthropic(
			model_name='claude-3-5-sonnet-20240620', timeout=100, temperature=0.0, stop=None
		),
	],
	ids=['gpt-4o-mini', 'gpt-4o', 'claude-3-5-sonnet'],
)
def planning_manager(request):
    """Initialize planning manager with different LLMs for testing"""
    return PlanningManager(llm=request.param)


def test_planning_system_prompt(planning_manager):
    """Test that system prompt is correctly formatted"""
    system_msg = planning_manager._get_planning_system_prompt()
    
    assert isinstance(system_msg, SystemMessage)
    assert "You are a task planning assistant" in system_msg.content
    assert "Analyze the given task description" in system_msg.content
    assert "reformulated task description" in system_msg.content
    assert "Estimated difficulty (1-10)" in system_msg.content


@pytest.mark.asyncio
async def test_plan_task_structure(planning_manager):
    """Test that plan_task returns correctly structured TaskPlan"""
    task = "Go to google.com and search for 'python programming'"
    
    task_plan = await planning_manager.plan_task(task)
    
    assert isinstance(task_plan, TaskPlan)
    assert isinstance(task_plan.task, str)
    assert isinstance(task_plan.short_summary, str)
    assert isinstance(task_plan.tags, list)
    assert all(isinstance(tag, str) for tag in task_plan.tags)
    assert isinstance(task_plan.estimated_difficulty, int)
    assert 1 <= task_plan.estimated_difficulty <= 10


@pytest.mark.asyncio
async def test_plan_task_content(planning_manager):
    """Test that plan_task generates meaningful content"""
    task = "Go to amazon.com and find the cheapest laptop under $500"
    
    task_plan = await planning_manager.plan_task(task)
    
    # Check task reformulation
    assert "amazon" in task_plan.task.lower()
    assert "laptop" in task_plan.task.lower()
    assert "$500" in task_plan.task
    
    # Check tags
    expected_tags = ["shopping", "amazon", "laptop", "price-comparison"]
    assert any(tag.lower() in [t.lower() for t in task_plan.tags] for tag in expected_tags)


@pytest.mark.asyncio
async def test_plan_task_error_handling(planning_manager):
    """Test error handling for invalid inputs"""
    with pytest.raises(ValueError):
        await planning_manager.plan_task("")
    
    with pytest.raises(ValueError):
        await planning_manager.plan_task(None)


@pytest.mark.asyncio
async def test_plan_task_difficulty_range(planning_manager):
    """Test that difficulty ratings are within expected range"""
    tasks = [
        "Go to google.com",  # Simple
        "Search for an image on google and download it",  # Medium
        "Compare prices of 5 different laptops across 3 different websites"  # Complex
    ]
    
    for task in tasks:
        task_plan = await planning_manager.plan_task(task)
        assert 1 <= task_plan.estimated_difficulty <= 10


