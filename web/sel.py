from selenium import webdriver
import time

driver = webdriver.Firefox("/home/anon/Programs/geckodriver-v0.30.0-linux64/")

driver.get("https://mail.protonmail.com/u/0/inbox")
time.sleep(5)
driver.find_element_by_id("username").send_keys("danieljosephjones")
driver.find_element_by_id("password").send_keys("qOdJQ@BqZd0s*EQvLLtyu28355f6c%8Lj4vpIHq^wboZ26qss#SsBCdjRbfOddCs")
time.sleep(30)
while True:
    selallbut = driver.find_element_by_id("idSelectAll")
    selallbut.click()
    time.sleep(2)
    driver.find_element_by_xpath("//body").send_keys("T")

