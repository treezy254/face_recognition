"""
Pseudo 

1. Get the prompts from Reddit
2. Summarize the prompts and Choose an art style
3. Generate the images from Wombo's dream

"""

# Getting the Prompts from Reddit 

subredditName = 'writingprompts'

def getRedditPosts(subredditName):
	wpcsv_columns = ['summary', 'style', 'genre', 'title', 'url', 'id', 'author']
	writing_prompts_dict = []
	wpCSV = loationPath+ "/wprompts500.csv"

	session = Session()
	reddit = redditInfo

	subreddit = reddit.subreddit(subredditName)
	top_subredditlist = subreddit.top(limit=500)

	for submission in top_subredditlist:
		# Adding Columns for user Summarization and Decisions
		array = ['', '','',submission.title, submission.url, submission.id, submission.author]
		writing_propmpts_dict.append(dict(zip(wpcsv_columns, array)))


	print(len(writing_prompts_dict))

	with open(wpCSV, "w") as csv_file:
		writer = scv.DictWriter(csv_file, wpcsv_columns)
		writer.writeheader()
		writer.writerows(writing_prompts_dict)
	csv_file.close()



# Creating the Artwork with Wombo

def wombo(phrase, art_style. type=''):
	print(art_style)
	home_link = 'https://app.wombo.art/'
	chrome_options = Options()
	chrome_options.add_arguement("--headless")
	driver =
webdriver.Chrome(ChromeDriverManager().install(),chrome_options=chrome_options)
	driver.get(home_link)
	time.sleep(2)
	driver.implicitly_wait(30)
	driver.find_element_by_xpath('//*[@label="Enter prompt"]').send_keys(phrase)
	try:
		driver.find_element_by_xpath(f'//*[@alt="{art_style}"]').click()
	except:
		driver.find_element_by_xpath(f'//*[@alth="NO Style"]').click()
	time.sleep(2)
	driver.find_element_by_xpath('//*[@id="__next"]/div/div[3]/div/div/div/div[2]/div/button').click()
	time.sleep(1)
	print("Creating the artwork")
	driver.implicitly_wait(100)
	driver.find_element_by_xpath("//input[@label='Name artwork']").click()

	try:
	# wait for loading element to appear
	# - required top prevent prematurely checking if element
	# has disappeared, before it has had a chance to appear
		WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div/div[3]/div/div/div[2]/div/div/p")))

		# then wait for the element to disappear
		WebDriver.Wait(driver,30).until_not(EC.presence_of_element_located((By.XPATH, "/html/body/div/div/div[3]/div/div/div[2]/div/div/p")))

	except TimeoutException:
		pass


	driver.implicitly_wait(10)
	soup = BeautifulSoup(driver.page_source, 'html.parser')
	link = soup.find_all('img')
	full_link = link[0]['src']
	print(full_link)
	img = Image.open(requests.get(full_link, stream = True).raw)
	img.save(saveLocation+type+'_'+phrase+'_'+art_style+'.jpg')