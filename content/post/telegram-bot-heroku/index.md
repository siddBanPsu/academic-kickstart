---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "COVID Information on Telegram Bot using Python, Git and Heroku"
subtitle: ""
summary: ""
authors: ["admin"]
tags: ["heroku", "telegram", "NLP"]
categories: []
date: 2020-04-24T22:44:01+08:00
lastmod: 2020-04-24T22:44:01+08:00
featured: true
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Using Telegram Bot to find COVID information."
  focal_point: "Center"
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
math: true
---

> **Okay, first things first.** If you want to check out the bot, please download [Telegram](https://telegram.org/) and then head to the [bot](https://t.me/covid_newsman_bot) directly.

*************

There is too much information on the web on COVID-19. However, I am very interested in primarily two major aspects:

1. Information on **Statistics** of each country. {{< hl >}}For example: deaths, new cases, recovered, etc.{{< /hl >}}
2. **Categorized news**, such that I can follow news specific to each topic. {{< hl >}}For example, new cases in countries, progress on drugs or vaccine, etc. {{< /hl >}}

The [Worldometer website](https://www.worldometers.info/coronavirus/#countries) provides amazing information on the first one. For the second, I found that [NewsApi.org](https://newsapi.org/) provides a great API (_maximum of 500 API Calls per day_) for free if your application is not used for commercial purposes.

### Main Steps ###

The main steps to get such a bot running are the following:

1. Set up Telegram Bot account. For details, go to this [page](https://core.telegram.org/bots) and directly to the section: **"How do I create a bot?"**
2. Create the python application for end-to-end updation of Worldometer and NewsApi data.
3. Pushing the code to Git and then automatically deploy. I used [Heroku](https://dashboard.heroku.com/) for deployment. 

I won't be describing Step 1 as it is pretty straightforward and all the information on how to set up a bot is available on Telegram.org's own website.
Let us focus on Step 2 and 3.

#### COVID application ####

The application needs to perform the following two tasks at regular intervals:

* Update information from **Worldometer** at regular intervals
* Update news regularly from **NewsApi**.

For Worldometer data, the process is pretty straightforward. I hit the website once every hour and update the information of all countries. Therefore, on the bot, there might be a difference than what the website shows. But, I bet that number will be small. I store the information as a Pandas dataframe using the following piece of code:

```python
def repeat_worldometer(COVID_URL):
      web_content = get_webpage_content(COVID_URL) # get the page source
      last_mod_time = str(get_last_modified_time(web_content)) # update when it was created last.
      worldometer_df = get_worldometer_df(web_content) # create the dataframe
      worldometer_df["mod_time"] = [last_mod_time]*worldometer_df.shape[0]
```

The dataframe contains two rows for each country: yesterday's statistics and current statistics.
Using this information, whenever any user requests the statistics using country name, the information is obtained from the dataframe and fed into a template that uses all the information to produce a formatted message. For example, when I type, _Singapore_ on the bot, I get the following output. **Pretty cool huh!**

{{< figure src="images/worldometer-message.png" title="Telegram bot showing Country Stats" lightbox="true" class="img-sm">}}

Serving this information using the bot api is pretty straightforward. Let us see what goes on inside.

```python
def show_country_stats(update, context):
    global c_map, worldometer_df
    try:
        country = str(update.message.text).lower() # what user enters
        closest_match_symspell = symspell_instance.lookup(country, verbosity=Verbosity.CLOSEST)
        if closest_match_symspell:
            closest_country = c_map[closest_match_symspell[0].term]
        else:
            context.bot.send_message(chat_id=update.effective_chat.id,
                        text="Sorry, the country you entered is not available:" + str(country))
            return
        todays_record, yesterdays_record = get_country_stats(worldometer_df, closest_country)
        reply = format_output(todays_record, yesterdays_record)
        # Send back the result
        context.bot.send_message(chat_id=update.effective_chat.id,
                                text=reply)
    except Exception as e:
        context.bot.send_message(chat_id=update.effective_chat.id,
                        text="Sorry, but the bot is not working now. It will be up and running soon...")
```

Wait. Why did the function _get_country_stats_ not use directly the _country_ entered by the user? I realized this when I started using the bot myself. I was making spelling mistakes. I typed _sngapore_ (missed the $i$) and did not get any results. That seemed to be an easy problem if I use a spelling corrector or do fuzzy match. I used SymSpell (an amazing toolkit for spelling correction) and basically find the closest match in the dictionary based on what the user enters. And after this, even when by mistake I search for _sngapore_ or _indea_, it still works. **Magic!!**

> Now, let's head to COVID related news. I am very interested to know what World leaders are talking about on COVID, or what vaccines are being created. I realized I had to search with keywords on Google every time. **Too much work!!**

I wanted to find news, and automatically tag them. Based on my previous experiences, [**Topic Modelling**](https://monkeylearn.com/blog/introduction-to-topic-modeling/), specifically Latent Dirchlet Allocation (LDA), seemed to be a great option. Get topic distributions of documents and group similar documents together. And when I try to look for documents with specific topics, just show me the recent ones on that topic. Easy peesy!!

I took several news documents retrieved from NewsApi and ran LDA on the text (title + description [only the first few words]). A big advantage of LDA: it is **unsupervised**. Therefore, I did not need to perform any annotations. I used a setting of 5 topics. The top 5 topics with the most important words along with their weights, are as follows:

```python
[(0,
  '0.026*"covid" + 0.026*"19" + 0.012*"positive" + 0.008*"tested" + 0.007*"test" + 0.006*"testing" + 0.006*"tests" + 0.006*"new" + 0.005*"cruise" + 0.005*"said"'),
 (1,
  '0.029*"cases" + 0.022*"new" + 0.020*"19" + 0.019*"covid" + 0.014*"news" + 0.009*"health" + 0.008*"positive" + 0.008*"death" + 0.008*"deaths" + 0.008*"state"'),
 (2,
  '0.023*"covid" + 0.023*"19" + 0.011*"pandemic" + 0.007*"people" + 0.006*"home" + 0.006*"help" + 0.006*"outbreak" + 0.006*"amid" + 0.004*"crisis" + 0.004*"fight"'),
 (3,
  '0.010*"pandemic" + 0.010*"businesses" + 0.008*"covid" + 0.008*"19" + 0.007*"amid" + 0.007*"business" + 0.007*"outbreak" + 0.006*"trump" + 0.006*"said" + 0.005*"relief"'),
 (4,
  '0.017*"covid" + 0.017*"19" + 0.014*"trump" + 0.011*"pandemic" + 0.009*"news" + 0.008*"outbreak" + 0.007*"spread" + 0.007*"world" + 0.006*"president" + 0.006*"health"')]
  ```

From the importance of the most important words, I assigned names to each of the topics:

* **COVID Progress + Research:** Information on vaccine/medications, etc.
* **Events:** Information on event cancellations, decisions, etc.
* **New Cases:** New cases in countries.
* **COVID Economy & Related:** Economic conditions, layoffs, etc.
* **COVID World news:** World leaders talking COVID.

This is how the bot looks with the created labels.

{{< figure src="images/news-categories.jpeg" title="Telegram Bot with News Categories" lightbox="true" class="img-sm">}}

The following piece of code shows the menu shown above. I used an additional category _Popular 10 Covid News_ that randomly selects 10 articles from the recent news that has been crawled.

```python
def show_news_options(update, context):
    keyboard = [[InlineKeyboardButton("Popular 10 Covid News", callback_data='-1')]] \
                + [[InlineKeyboardButton(x, callback_data=str(i))] for i, x in enumerate(news_topics)]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Please choose from the following options:', reply_markup=reply_markup)
    update.message.reply_text('You can type country names for statistics if you are interested in Worldometer statistics instead. For example: usa, india, etc. Type /start anytime to see main menu')
```

Clicking on any of the topic will use a callback function to get new documents of the specific topic. For new news artiles, the topic is assigned based on the topic that has the maximum contribution based on word-distributions. I also assigned a threshold such that documents without enough evidence on any topic do not show up. **Relevance is important!**

When I select the button on **COVID Research**, I receive the following articles. Pretty relevant :)

{{< figure src="images/news-research.jpeg" title="Telegram Bot showing articles on COVID Research" lightbox="true" class="img-sm">}}

Such category-specific information is very helpful for me and I can choose what topics of articles to read now.

In addition to category-specific information, I have allowed search on news articles too. For example, using _/search lysol_ to get information on lysol (*blink blink*), I get pretty relevant news (really????) to read.

{{< figure src="images/news-lysol.jpeg" title="Telegram Bot showing articles on Lysol ;)" lightbox="true" class="img-sm">}}

In order to repeatedly get new news articles, I used an amazing library called [_schedule_](https://pypi.org/project/schedule/) in python.

#### Deploying on Heroku ####

The deployment part is pretty straightforward. The github repo can be directly linked to have continuous development on Heroku. First, create an application on Heroku. In the root folder of the application, a Procfile needs to be created, only with the following content where telebot.py is the main python file to run. You have to set up automatic deploys on Heroku.

```c++
worker: python app/telebot.py
```

There are several types of dynos on Heroku giving several types of benefits, starting from a Free Tier. If you need some minimal level of tracking and metrics, a Hobby Dyno ($7 per month) is a good to have.

*********

I hope the above tutorial gave a decent overview on creating a Telegram bot with a certain functionality (that I find really helpful).

**I plan to release the code soon!**