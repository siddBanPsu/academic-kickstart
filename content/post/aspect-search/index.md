---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Aspect Based Semantic Review Search"
subtitle: ""
summary: ""
authors: [Siddhartha Banerjee]
tags: ["Deep Learning"]
categories: []
date: 2020-01-28T22:44:01+08:00
lastmod: 2020-01-28T22:44:01+08:00
featured: false
draft: true

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "User searches for reviews related to location of the stay."
  focal_point: "Center"
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []

---

Have you ever tried searching Airbnb reviews to learn about specific characteristics of an accomodation you are trying to book? The current system simply allows for keyword-based search. However, keyword-based search is seldom exhaustive and does not show the best possible results to the end-user. 

Consider the example in the review search above. The user is interested to answer the following question: _How is the location of the listing?_ As shown in the search results, the word **location** is highlighted and it is basically keyword-based search. Some minor variations might also be searchable; for example, _located_. However, there can be multiple reviews that imply talking about the location of the listing without ever mentioning the word _location_. Can we perform a better search? Let's try. 

To understand searchability of reviews, we will rely on open datasets published by [AirBnb on Kaggle](https://www.kaggle.com/airbnb/seattle). As mentioned earlier, our goal is to search reviews beyond using keywords. This is related to the concept of **aspects** as commonly used in literature associated with [Sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis). Aspects are attributes of an entity. In this case, _location_ is an attribute of the accomodation listing. 

I came across this paper: [An Unsupervised Neural Attention Model for Aspect Extraction](https://www.aclweb.org/anthology/P17-1036/) published in ACL 2017. The title of the paper implied we do not need actual labels of aspects and still be able to aspect extraction. While similar goals can be achieved using [Latent Dirchlet Allocation or LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation), comparisons in the paper show that the Neural Attention model is far superior than LDA or similar methods. Therefore, I decided to explore the Neural Attention Model. My plan was to automatically infer aspects, and use the aspects to guide the search based on similarity scores. 

#### Neural Attention Model for Aspect Extraction ####

The idea of the paper is pretty straightforward. The author hypothesizes that an aspect will have a feature representation (embedding) that would be close to the words that explain the aspect.








