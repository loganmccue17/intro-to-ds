---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
## Web-Scraping Presidential Information

An demonstration of web-scraping and condensing gathered information into a data table. Further analysis is done exploring demographic information on each president. <br>
**Note: Conclusions Need to Be Revised Accounting for the Majority of Popular Vote Without a Simple >=50% Conditional**

# Are Presidential Popular Vote Wins Characterized by Political Party or Religion?

+++

We begin our analysis by importing libraries from pandas, seaborn, and BeautifulSoup

```{code-cell} ipython3
import requests
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

'requests' allows us to request access when we webscrape from various websties. BeautifulSoup is the package that allows us to webscrape in general and gives us tools when doing it. Pandas is needed for data manipulation and seaborn is needed for data visualization.

+++

First we save the url for presidential information from the millercenter. The millercenter is an affiliate of the University of Virginia which deals in political history. We then use requests and BeautifulSoup to get the htmlfile and then parse it to store the html code as the variable miller_presidents

```{code-cell} ipython3
miller_presidents_url = 'https://millercenter.org/president'
```

```{code-cell} ipython3
miller_presidents_html = requests.get(miller_presidents_url).content

miller_presidents = BeautifulSoup(miller_presidents_html,'html.parser')
```

I will not output the full html code as it is quite long. But miller_presidents essentially stores the html code for the website.

+++

We then inspected the html code for the website to find the segments pertaining to each individual president. This was under the html title "div" and under the class "president-info views-row". Using the .find_all() command, we are allowed to separate each instance of this and store the result in the list 'presidents'. Afterwards we printed the length of 'presidents' to confirm.

```{code-cell} ipython3
presidents = miller_presidents.find_all("div", "president-info views-row")
```

```{code-cell} ipython3
len(presidents)
```

As we can see, there are 46 items in the 'presidents' list, which makes sense as there have been 46 presidents. This proves that the separation of all presidents on this homemade site was successful.

+++

We then printed the first instance in presidents: 'presidents[0]'. This should be a generic code about George Washington, with a link that will direct us to more information on George Washington.

```{code-cell} ipython3
presidents[0]
```

As we can see, this is a much shorter sample of code, all directed on a few details of George Washington.

+++

We then parse the html file code to get a string of the name of the president: George Washington. Looking at the html code under presidents[0], we see that the string "George Washington" is with a "p" header, under the class "views-field--title". We use a .find() function to look for that instance. We then notice that it is further located under an "a" tab as a string. The final code to end up with the name of the president is shown below:

```{code-cell} ipython3
presidents[0].find("p", "views-field--title").a.string
```

As we can see this worked for George Washington and it can be assumed that this will work for every president as the format of the website looks generally the same.

+++

We can test that theory my setting up a for loop for every person in the presidents list. We will attempt to find the name of the president in the same location in the html file and store our president names to a respective list.

```{code-cell} ipython3
president_names = [person.find("p", "views-field--title").a.string for person in presidents]
president_names
```

And as we can see, that worked. We have a list of every president in order from George Washington to our current president, Joe Biden.

+++

Next we will use a very similar method to obtain a link each president's specific webpage. This is located in the same "a" header as before, but this time we look for an attribute of the "a" header that has 'href'. This will give us our link for George Washington.

```{code-cell} ipython3
washington_url = presidents[0].find("p", "views-field--title").a.attrs['href']
washington_url
```

Now while this link is correct, it is only an extension from the greater millercenter.org website. So when we use requests and BeautifulSoup to parse and obtain our html information, we must concatenate "https://millercenter.org" with our washington_url extension. After doing that we get the html information for George Washington.

```{code-cell} ipython3
washington_html = requests.get("https://millercenter.org" + washington_url).content
washington_info = BeautifulSoup(washington_html,'html.parser')
```

Similarly, I will not print out the full washington_info variable which stores our information, but with subsequent steps we can see that it worked.

+++

For instance, our next step is looking for the fast facts on George Washington. On the website itself, these appear in their own separate box and we need to find that box. Looking at the html code, we notice that all of the facts are within the "div" heading under the "fast-facts-wrapper" class. We can use the .find() function to look for that section on George Washington.

```{code-cell} ipython3
washington_info.find("div", "fast-facts-wrapper")
```

We can see that it was successfully printed! This fast facts section has great information on their start and end date, political affiliations, birth place, marriage information, and more!

+++

Perhaps the most useful information for our purposes is their political affiliation. We can look for it by specifically using the .find() function to look for a heading "label" in which the string portion of that label is "Political Party". As this only appears once, we can get our political_party_label. However, this is not enough. We use the .find_next_sibling() function with the "div" parameter to look for the "div" section that appears AFTER this label. We strip this text to obtain our political party.

```{code-cell} ipython3
political_party_label = washington_info.find("label", string = "Political Party")
political_party_info = political_party_label.find_next_sibling("div").text.strip()
political_party_info
```

As we can see, the political party affiliation for George Washington is a "Federalist". This is exactly what we are looking for and matches what we expect from George Washington. Though he frowned against political parties, his views aligned more with the Federalists.

+++

Since we expect the method of obtaining this string to be the same among all presidents, we can run a loop that obtains the parties for each president. This uses the exact same method as before. As we expect this to work, we create a list for not only the parties, but for other key data as well. In specific, we obtain presidential birth_places, inauguration_dates, end_dates, educations, religions, and presidential order. These are all separate lists that we plan to use in a dataframe later.

```{code-cell} ipython3
parties = []
birth_places = []
inauguration_dates = []
educations = []
religions = []
end_dates = []
pres_order = []

for president in presidents:
    president_url = president.find("p", "views-field--title").a.attrs['href']
    president_html = requests.get("https://millercenter.org" + president_url).content
    president_info = BeautifulSoup(president_html,'html.parser')

    # Search and store parties
    party_label = president_info.find("label", string = "Political Party")
    political_party = party_label.find_next_sibling("div").text.strip()
    parties.append(political_party)

    # Search and store birthplaces
    birthplace_label = president_info.find("label", string = "Birth Place")
    birthplace = birthplace_label.find_next_sibling("div").text.strip()
    birth_places.append(birthplace)

    # Search and store inauguration dates
    inauguration_label = president_info.find("label", string = "Inauguration Date")
    inauguration = inauguration_label.find_next_sibling("div").text.strip()
    inauguration_dates.append(inauguration)

    # Search and store education info
    education_label = president_info.find("label", string = "Education")
    try:
        education_level = education_label.find_next_sibling("div").text.strip()
        educations.append(education_level)
    except:
        educations.append(pd.NA)

    # Search and store religion info
    religion_label = president_info.find("label", string = "Religion")
    religion = religion_label.find_next_sibling("div").text.strip()
    religions.append(religion)

    # Search and store term end date info
    term_end_label = president_info.find("label", string = "Date Ended")
    try:
        end_date = term_end_label.find_next_sibling("div").text.strip()
        end_dates.append(end_date)
    except:
        end_dates.append("Present")

    # Search and store president number
    pres_number_label = president_info.find("label", string = "President Number")
    pres_number = pres_number_label.find_next_sibling("div").text.strip()
    pres_order.append(pres_number)

religions
```

As an example, we printed out the presidental religions. We can see that this worked for religions, and without errors, for all of our other lists. This is great as we can now create a dataframe summarizing our new dataset!

+++

We create the dataset here by using the pandas DataFrame method. We pass in a dictionary that contained a column title as our first element and the respective list as our second element. This creates a dataframe in nice organized columns

```{code-cell} ipython3
president_df = pd.DataFrame({'Presidential Order' : pres_order,'Name':president_names,'Party':parties,
              'Birth Place':birth_places, 'Inauguration':inauguration_dates,
                             'End Date':end_dates, 'Education':educations, 'Religion':religions})

president_df.head()
```

We can see our nice dataframe here using the president_df.head() method. We can see each president's party, name, religion, among others! Webscraping this website has worked!

+++

However, now we want continuous variables to analyze. We choose the public website 'britannica' which has data on every presidential elections in history! This dataset includes the year, the winner, the electoral votes granted, the popular votes granted (when applicable) and the popular percentage won (when applicable). We begin our analysis by similarly requesting the html file and storing the parsed data in a variable called election_data.

```{code-cell} ipython3
elections_url = 'https://www.britannica.com/topic/United-States-Presidential-Election-Results-1788863'
elections_html = requests.get(elections_url).content
election_data = BeautifulSoup(elections_html,'html.parser')
```

I will not print out this election_data variable as it is quite large, but we will soon see that the request to access the html file had worked.

+++

We first need to separate each election and its data. We do this with another find_all function which looks for the heading "tr" and the class "has-rs". We knew to use these parameters by looking at the html file for this website and seeing where each election was separated.

```{code-cell} ipython3
elections = election_data.find_all('tr', 'has-rs')
```

```{code-cell} ipython3
len(elections)
```

```{code-cell} ipython3
elections[0]
```

We can see that each election is properly stored in a list by examining the first element which only has information on the 1789 'election'. We also use the len() function to see how many elections are within the list. Since this numnber is 59, which is accurate, we can move on to the next step.

+++

We then look for the string that has our election year. We can see that luckily it is the only string with our first "a" header of elections[0]. Using this knowledge, we can simply find our election year by adding the .a.string subtag.

```{code-cell} ipython3
elections[0].a.string
```

This worked. Given that information, we can try using the same method with for the other bits of information that we want to gather.

+++

We noticed that .a.string will clearly not work for each piece as there is a different address within the html file. The other bits of information are under the 'td' tag with a value in brackets like [1] succeeding the tag to specify which one. We can then use contents[0] to grab the portion that we want.

```{code-cell} ipython3
years = []
names = []
electoral_votes = []
popular_votes = []
popular_percentage = []

for election_cycle in elections:
    years.append(election_cycle.a.string)
    names.append(election_cycle.find_all('td')[1].a.string)
    electoral_votes.append(election_cycle.find_all('td')[3].contents[0].strip())
    try:
        popular_votes.append(election_cycle.find_all('td')[4].contents[0].strip())
        popular_percentage.append(election_cycle.find_all('td')[5].contents[0].strip())
    except:
        popular_votes.append(pd.NA)
        popular_percentage.append(pd.NA)

names
```

We can see that this worked for all of them! The key difference other than the location is that both the popular votes and popular vote percentage is not always given for an elections (pre-JQA). This means that these variables need to be used under a try/except keyword that adds the given value and if not available, gives us NA.

+++

To visualize this information, we would like to once again convert the data into a dataframe. We do this using pandas' DataFrame method which takes in a dictionary as its parameter. The dictionary contained the column name we would like as the first element and then the list as the second element once again. To make later calculations easier, we then delete commands that represent 'thousands' and then convert the electoral votes and popular votes to an integer and the popular percentage to a float.

```{code-cell} ipython3
elections_df = pd.DataFrame({'Years' : years,'Name':names,'Electoral Votes':electoral_votes,
              'Popular Votes':popular_votes, 'Popular Percentage':popular_percentage})

elections_df['Electoral Votes'] = elections_df['Electoral Votes'].str.replace(',', '').astype('Int64')
elections_df['Popular Votes'] = elections_df['Popular Votes'].str.replace(',', '').astype('Int64')
elections_df['Popular Percentage'] = elections_df['Popular Percentage'].astype('Float64')

elections_df.head()
```

This is our resultant dataframe (note that the head contains NA values for popular votes and popular percentages. This is because the popualar vote was not recorded before John Quincy Adams.

+++

We then create a new dataframe called complete_pres_df which merges these two datasets (one from millercenter and the other from britannica). We merge them with a pandas function and use the on = 'Name' parameter so that for each election, we also provide the information for a given president. This could be cleaned up later to have minimal duplication, but it works for our purposes effectively.

```{code-cell} ipython3
complete_pres_df = pd.merge(elections_df, president_df, on = 'Name')

complete_pres_df.head()
```

We can see that this worked well for us and now we have the merged voting and president fact data!

+++

Now before we begin our exploratory analysis, there is one last piece of data I would like to gather. We have the total popular votes and the percent of the popular votes won. I would like to also obtain a percent of the electoral votes won so we can compare this later. To do so, I need the total electoral votes for each elections (as I already have how many electoral votes each candidate won). To obtain this, we are going to use the pandas method read_html()

```{code-cell} ipython3
electoral_votes = pd.read_html('https://en.wikipedia.org/wiki/List_of_United_States_presidential_elections_by_Electoral_College_margin', header = 0)[2]
```

This gives us a list of dataframes within the html file provided by the link, which is a wikipedia article.

+++

We will choose the dataframe provided by the index [2:]. We can then select the columns that we need and sort the values based on the 'Year' column to get our table in chronilogical order. 

```{code-cell} ipython3
election_total_df = electoral_votes[2:][['Year', 'Number of electors voting']].sort_values(by = 'Year')
election_total_df.head()
```

This is close to what we want, but it still needs some cleaning for us to use it.

+++

We can start by replacing the year at index 60, with 1789 (instead of the awkward range). We can then remove the [b] subtag for the number of electors voting at index 5. These are luckily the only two odd inputs that need to be manually changed. Asides from that, we need to reset the index of the dataframe so that everything is now in its proper order.

```{code-cell} ipython3
election_total_df.loc[5, 'Number of electors voting'] = '138'
election_total_df.loc[60, 'Year'] = 1789

election_total_df = election_total_df.reset_index()
election_total_df.head()
```

This dataframe is still in need of cleaning, but given that the 'Year' column in this new dataframe is in matching order to our original dataframe, it works for what we need.

+++

So we will get the 'Number of electors voting' column, string strip the value, and convert into the 'Int64' type so we can perform future calculations. We add this column to our dataframe and then perform a calculation based on two columns to calculate our 'Electoral Percentage' that was won.

```{code-cell} ipython3
complete_pres_df['Total Electors'] = election_total_df['Number of electors voting'].str.strip().astype('Int64')
complete_pres_df['Electoral Percentage'] = (complete_pres_df['Electoral Votes'] / complete_pres_df['Total Electors']) * 100
complete_pres_df.sample(5)
```

As we can see, we have one column based on our newest imported dataframe and then a second column that denotes the percent of electoral college votes won.

+++

Furthermore, we can do two new calculations that will be interesting for out study. The first is the percent difference between the popular percentage won and the electoral college percentage won. Essentially, this will demonstrate how much greater the electoral college was won than the popular vote. The second is a simple boolean based on whether the percent the popular vote was won is greater than 50%. This simply tells us if a president won the popular vote.

```{code-cell} ipython3
complete_pres_df['Percent Difference from Popular Percentage'] = ((complete_pres_df['Electoral Percentage'] 
                                                                     - complete_pres_df['Popular Percentage']) 
                                                                        / complete_pres_df['Popular Percentage']) * 100

complete_pres_df['Win Popular?'] = complete_pres_df['Popular Percentage'] > 50
complete_pres_df.sample(5)
```

As we can see, these two columns were successfully added!

+++

We begin our exploratory data analysis by simply checking how many of our presidents (who all won the electoral college vote), also won the popular vote. This is done by doing a simple value_counts() of our 'Win Popular?' columns.

```{code-cell} ipython3
complete_pres_df['Win Popular?'].value_counts()
```

What we can see is that about half of our elected presidents won the popular vote, meaning that for only half of our presidents did a majority of citizens actually vote for that person. This provides some insight on how drastically different the popular votes and electoral votes for a president can be.

+++

Now I would like to explore the presidents further based on whether or not they won the popular vote. This may provide further insight on why this is the case. First, we will look at summary statistics for Percent of the Electoral College won, based on whether or not they won the popular vote.

```{code-cell} ipython3
complete_pres_df.groupby(['Win Popular?'])['Electoral Percentage'].describe()
```

We can see that the average percent of the electoral college won is about 75% for presidents that also won the popular vote. This average drops to just under 60% for presidents that lost the popular vote. This is a pretty significant difference, implying that the presidents that did not win the popular vote, usually only won the electoral college but a slight margin. However this is not always the case as the maximum electoral college percentage won for a president who did not win the popular vote is just over 80%, which is a significant majority.

+++

Now lets use the same categorical variable: whether or not the president won the popular vote. This time, however, we will look at summary statistics for the percent difference that the electoral college was won over the popular vote.

```{code-cell} ipython3
complete_pres_df.groupby(['Win Popular?'])['Percent Difference from Popular Percentage'].describe()
```

Using these summary statistics, we can see that for presidents that did not win the popular vote, they won the electoral college by a percentage about 21% on average higher than they did the popular vote. For the presidents that did win the popular vote, they won the electoral college by a percentage about 37% on average higher than they did the popular vote. These seem to be significant margins at first glance, but the standard deviation of each group implies that there is a lot of variation in this statistic. This is compared to the electoral percentage by itself, which has a much lesser range. This implies that that statistic holds greater pattern than the percent difference from popular percentage. However, the data suggests both times that the presidents who did not win the popular vote, usually only won the presidency narrowly (at least more narrowly than presidents who won the popular vote).

+++

Next we attempt to see if any other factors hold patterns on whether a president won the popular vote or not. We first look at religion, where we expect no significant results.

```{code-cell} ipython3
complete_pres_df.groupby(['Religion'])['Win Popular?'].value_counts()
```

It is interesting to point out that some patterns exist when examining religion's effect on the popular vote. For instance, we can see that for the 14 Presbyterian presidents that we have had, a significant 10 of them did not win the popular vote, which is far from a 50-50 split. This is similar for Episcopalian and Methodists, where 5 presidents won the popular vote for each, but much less did not win the popular vote. Lastly, it is important to note Baptists, where four of them did not win the popular vote, and only one did. While this could be out of randomness, it would be interesting to do a separate analysis of these religions and these presidents to see if there are trends in the religion's popularity, scandals, or the time period as a whole that explain this.

+++

What is expected to be more significant is the impact on a president's political party on whether or not they won the popular vote. While expected to be more significant than religion as a factor, one would still expect not a drastic effect.

```{code-cell} ipython3
party_v_popular = complete_pres_df.groupby(['Party'])['Win Popular?'].value_counts().reset_index()
party_v_popular
```

However, some interesting results do come from this analysis. While the data could be cleaned more to group certain sub-parties together, like "Democratic" and "Democrat", the data does presdent an interesting result. For Democrats, there is about a 50-50 split on whether or not a president won the popular vote or not. Surprisingly, for the Republicans, while 11 of them won the popular vote, only 5 of them did not. While this is only a preliminary analysis, it is interesting to see this biasy. This could be due to the Republicans' overall popularity post-Civil War, but it would be interesting to sort this data into 50-year buckets to examine these patterns.

+++

Next we are going to select only specific rows of the data and combine the Party and Win Popular? columns. This way we can easily graph these four primary scenarios to better visualize our results.

```{code-cell} ipython3
party_v_popular['Party_Popular'] = party_v_popular['Party'] + party_v_popular['Win Popular?'].astype(str)
party_v_popular = party_v_popular.iloc[[0,1,4,5]]
party_v_popular
```

That worked. Just to practice data visualization, we will graph this is a barplot using seaborn.

```{code-cell} ipython3
sns.barplot(data = party_v_popular, x = 'Party_Popular', y = 'count')
```

Perfect! This is just a very basic plot, but it visualizes the conclusions we have already made while showing the significance in the 'RepublicanFalse' category.

+++

For our last exploratory data analysis, we will visualize the data from the religion conclusion as well, again using seaborn.

```{code-cell} ipython3
religion_v_popular = complete_pres_df.groupby(['Religion'])['Win Popular?'].value_counts().reset_index()
religion_v_popular['Party_Popular'] = religion_v_popular['Religion'] + religion_v_popular['Win Popular?'].astype(str)
religion_v_popular = religion_v_popular.iloc[[0,1,7,8,9,10,13,14]]
sns.barplot(data = religion_v_popular, x = 'Party_Popular', y = 'count')
plt.xticks(rotation=45)
```

I had to look up how to tilt the x-axis labels using matplotlib, but this is visualization based on data we have already examined. This models religions effect on popular vote (for the main four religions that presidents have had). While interesting data, more data analysis could be performed to better examine this.

+++

This is enough data examination for this assignment. We have successfully loaded in data sets from three sources. The first was a webscraping of millercenter, which required us looping through sub-websites linked from the homepage of our website. The second was from webscraping a single britannica site. The third source was from wikipedia and was loaded through read_html(). This was done only to get the total electoral college votes for each election. We have examined interesting results comparing the popular vote and the electoral college vote for each election. We have also examined the potential impact that religion and political affiliation have had on this data. While this could be random results for those latter analyses, it would be interesting to expand this exploration to examine these topics! The final thing we will do is save our full dataset a a csv, called complete_presidential.csv

```{code-cell} ipython3
complete_pres_df.to_csv('complete_presidential.csv')
```
