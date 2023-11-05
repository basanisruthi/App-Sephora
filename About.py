import streamlit as st
import pandas as pd
st.set_page_config(page_title="About", page_icon='📊')

st.header("📊 About The Project")

st.markdown("---")

st.markdown("### 🎯 Goal :")

st.write("""Sephora stands as the preeminent luxury goods conglomerate globally, boasting best-in-class e-commerce experiences. Beside reaching customers through channels such as Facebook Live Shopping, and Instagram Checkout, Sephora has also prioritised digital means such as integrating in-store technologies to engage clients, and personalizing product or service recommendations based on customer data. Furthermore, in today’s social media fueled world, global brands tag an additional premium to monitor reviews and feedback and quickly act on them. This ability to turnaround the feedback gives brands an edge over their competitors. However, given the large number of products offered by Sephora, and its global presence, this means that Sephora must sieve through large swaths of data and get them to the relevant departments.  
As such, in alignment its core values and business trajectory, Sephora should leverage a data-driven approach, applying text analytics to sort through large amounts of reviews and feedback, so the management can strive to enhance customer satisfaction and foster greater customer loyalty. The intent would be to channel these reviews to the relevant departments for them to act upon them, improving the turnaround time experienced by the customer.""")

    
st.markdown("### 🔬 Project Overview (for the scope of this App :)")

st.write("""This project seeks to establish an advanced review and feedback system tailored for Sephora, designed to automate menial processes, and extract valuable business insights. This would be achieved through  the following """)

st.markdown("- (1) tagging of after-sales product reviews for action by the relevant departments (LDA)")
st.markdown("- (2) assess the sentiments expressed in reviews, providing Sephora with a greater understanding of customer emotions and satisfaction levels (Naïve Bayes)")
st.markdown("- (3) continuously monitor feedback and identify overarching trends within these reviews (Topic modelling -  LDA)")

st.write("""By amalgamating these models into a working dashboard, Sephora executives will be provided with a means of understanding the macro view of sentiments of its products, while being able to drill down to specific brand comparison, and customer facing performance of each functional department.""")

st.markdown("### 😀 Note to the User")

st.write(""" This app has two pages """)

st.markdown(" First Page: Sentiment analysis using VADER & Textblob")

st.markdown("- The first page to predict sentiment and its intensity Using VADER to predict the sentiment and Textblob to preprocess the data. However,Because VADER is specifically tuned for social media text, it can often perform well on informal and short text messages like those found on platforms like Twitter.Keep in mind that while VADER can be useful only for certain applications, it may not always be as accurate or nuanced as more advanced models, especially for longer or more complex texts ")

st.markdown(" Second Page: Master Dashboard")
st.markdown("- The Second Page, called 'Master Dashboard', assists the user in labeling reviews as either positive or negative.")
st.markdown("- To utilize this feature, the user should follow these steps:")
st.markdown("  - Select 'On CSV' from the drop-down menu on the left-hand side.")
st.markdown("  - Proceed to upload the file.")
st.markdown("- Once the reviews are labeled, the final result can be downloaded by the user in CSV format.")


