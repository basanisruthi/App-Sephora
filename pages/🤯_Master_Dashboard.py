import streamlit as st
import gensim
from gensim import corpora, models
import re
import os
import shutil
import heapq
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#### topic labeling ##########

product_keywords = ['cream', 'foundation', 'lipstick', 'perfume', 'makeup', 'acne', 'irritated skin', 'worst product', 'bad product', 'poor quality', 'harsh', ' harsh chemicals', 'toxic chemicals', 'greasey', 'hand cream', 'oily', 'break out', 'cured acne', 
                    'smooth', 'soft','recommend','moisterized', 'must have','sunscreen','lotion','dissapointed','redness','allergic','bronzer','lipbalm','lips']

marketing_keywords = ['good deal', 'gift set', 'promotions', 'festive offer', 'end of season sale', 'bundle offer', 'dicounts', 'cupon code','hype','marketing gimmick','expectations','not worth','christmas gift','gift']

last_mile = ['Late delivery', 'service', 'poor service', 'delivery agent rude', 'horrible delivery', 'quick delivery', 'ontime', 'very late', 'poor delivery','experience', 'not happy', 'satistifed', 'delivery', 'very rude','mishandled','marked as delivered', 'not delivered']

packing = ['Love packaging', 'pretty packaging', 'packaging is very bad', 'foul selling package', 'love package', 'tube','broken','pump','inadequate packaging', 'crushed','damaged']


industries = {
    'Product Team': product_keywords,
    'Marketing Team': marketing_keywords,
    'Logistics Team': last_mile,
    'Packing Team': packing
}

# Function to transform text using CountVectorizer
def transform_text(text):
    return vectorizer.transform([text])

# Function to predict sentiment using Naive Bayes
def predict_sentiment(text):
    input_vector = transform_text(text)
    prediction = naive_bayes.predict(input_vector)
    return prediction[0]

def label_topic(text):
    """
    Given a piece of text, this function returns the top two departments labels that best match the topics discussed in the text.
    """
    # Count the number of occurrences of each keyword in the text for each industry
    counts = {}
    for industry, keywords in industries.items():
        count = sum([1 for keyword in keywords if re.search(r"\b{}\b".format(keyword), text, re.IGNORECASE)])
        counts[industry] = count
    
    # Get the top two industries based on their counts
    top_industries = heapq.nlargest(2, counts, key=counts.get)
    
    # If only one industry was found, return it
    if len(top_industries) == 1:
        return top_industries[0]
    # If two industries were found, return them both
    else:
        return top_industries

def preprocess_text(text):
    # Replace this with your own preprocessing code
    # This example simply tokenizes the text and removes stop words
    tokens = gensim.utils.simple_preprocess(text)
    stop_words = gensim.parsing.preprocessing.STOPWORDS
    preprocessed_text = [[token for token in tokens if token not in stop_words]]

    return preprocessed_text

def perform_topic_modeling(transcript_text,num_topics=1,num_words= 5):
    """
    this function performs topic modelling on a given text.
    """
    preprocessed_text = preprocess_text(transcript_text)
    # Create a dictionary of all unique words in the transcripts
    dictionary = corpora.Dictionary(preprocessed_text)

    # Convert the preprocessed transcripts into a bag-of-words representation
    corpus = [dictionary.doc2bow(text) for text in preprocessed_text]

    # Train an LDA model with the specified number of topics
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
     # Extract the most probable words for each topic
    topics = []
    for idx, topic in lda_model.print_topics(-1, num_words=num_words):
        # Extract the top words for each topic and store in a list
        topic_words = [word.split('*')[1].replace('"', '').strip() for word in topic.split('+')]
        topics.append((f"Topic {idx}", topic_words))

    return topics

# Load data for sentiment analysis (assuming df is your DataFrame)
df = pd.read_csv("Sentiment Analysis.csv")
df.columns = ["Text", "Label"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Text'])

naive_bayes = MultinomialNB()
naive_bayes.fit(X, df['Label'])

st.set_page_config(layout="wide")

choice = st.sidebar.selectbox("Select your choice", ["On Text","On CSV"])

if choice == "On Text":
    st.subheader("Topic Modeling, Sentiment Analysis, and Labeling")
    
    text_input=st.text_area("Paste your input text",height = 400)
    
    if text_input is not None:
        if st.button("Analyze Text"):
            col1, col2, col3,col4 =st.columns([1,1,1,1])
            with col1:
                st.info("Text is below")
                st.success(text_input)
            with col2:
                topics= perform_topic_modeling(text_input)
                st.info("Topics in the text")
                for topic in topics:
                    st.success(f"{topic[0]}: {', '.join(topic[1])}")
            with col3:
                st.info("Sentiment Analysis")
                sentiment = predict_sentiment(text_input)
                if sentiment == 1:
                    st.markdown("Sentiment: Positive 😃")
                else:
                    st.markdown("Sentiment: Negative 😠")
            with col4:   
                st.info("Topic Labeling")
                labeling_text = text_input
                industry = label_topic(labeling_text)
                st.markdown("**Topic Labeling Industry Wise**")
                st.write(industry)
               
            
elif choice == "On CSV":
    st.subheader("Topic Modeling and Labeling on CSV File")
    upload_csv = st.file_uploader("Upload your CSV file", type=['csv'])
    if upload_csv is not None:
        if st.button("Analyze CSV File"):
            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                st.info("CSV File uploaded")
                csv_file = upload_csv.name
                with open(os.path.join(csv_file),"wb") as f: 
                    f.write(upload_csv.getbuffer()) 
                print(csv_file)
                df = pd.read_csv(csv_file, encoding= 'unicode_escape')
                st.dataframe(df)
            with col2:
                data_list = df['Data'].tolist()
                industry_list = []
                sentiment_list = []  # Added to store sentiment results
                topic_list = []  # Added to store topics
                for i in data_list:
                    industry = label_topic(i)
                    industry_list.append(industry[0])  # Extract the first element from the list
                    
                    # Perform sentiment analysis
                    sentiment = predict_sentiment(i)
                    sentiment_label = "positive" if sentiment == 1 else "negative"
                    sentiment_list.append(sentiment_label)
                    
                    # Perform topic modeling
                    topics = perform_topic_modeling(i)  # Assuming you have a function for topic modeling
                    topic_info = '\n'.join([f"Topic {topic[0]}: {', '.join(topic[1])}" for topic in topics])  # Create topic info string
                    topic_list.append(topic_info)
                    
                df['Industry'] = industry_list
                df['Sentiment'] = sentiment_list  # Added Sentiment column
                df['Topics'] = topic_list  # Added Topics column
                st.info("Topic Modeling and Labeling")
                st.dataframe(df)
                
                # Add a download button for the modified CSV
                st.download_button(label="Download labled CSV", data=df.to_csv(), key='download_button', file_name='modified_data.csv')
                # Add pie chart below the DataFrame
                import plotly.express as px
                sentiment_counts = df['Sentiment'].value_counts()
                fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index,
                             title='Overall Positive and Negative Reviews Ratio')
                st.plotly_chart(fig)
                
                # Add bar chart for count of positive and negative reviews by industry
                industry_sentiment_counts = df.groupby(['Industry', 'Sentiment']).size().reset_index(name='Counts')
                fig2 = px.bar(industry_sentiment_counts, x='Industry', y='Counts', color='Sentiment',
                              labels={'Counts': 'Number of Reviews'}, barmode='group',
                              title='Positive and Negative Reviews by Departments')
                st.plotly_chart(fig2)
