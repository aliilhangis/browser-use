import time

from tokencost import count_string_tokens

from browser_use.agent.views import AgentAction
from browser_use.browser.service import Browser
from browser_use.dom.service import DomService
from browser_use.utils import time_execution_sync


# @pytest.mark.skip("slow af")
def test_process_html_file():
	browser = Browser(headless=False)

	driver = browser._get_driver()

	dom_service = DomService(driver)

	driver.get('https://kayak.com/flights')
	# browser.go_to_url('https://google.com/flights')

	time.sleep(2)
	# browser._click_element_by_xpath(
	# 	'/html/body/div[5]/div/div[2]/div/div/div[3]/div/div[1]/button[1]'
	# )
	# browser._click_element_by_xpath("//button[div/div[text()='Alle akzeptieren']]")

	elements = time_execution_sync('get_clickable_elements')(dom_service.get_clickable_elements)()

	print(elements.dom_items_to_string(use_tabs=False))
	print('Tokens:', count_string_tokens(elements.dom_items_to_string(), model='gpt-4o'))

	input('Press Enter to continue...')
