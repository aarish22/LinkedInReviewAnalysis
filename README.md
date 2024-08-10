# LinkedIn Reviews Analysis

This project involves an exploratory data analysis (EDA) and sentiment analysis of LinkedIn reviews. The aim is to understand the distribution of ratings, the sentiment of the reviews, and to visualize the key terms associated with different sentiments.

## Dataset

The dataset used in this project is `linkedin-reviews.csv`, which contains the following columns:

- **Review**: The text of the review left by a user.
- **Rating**: The numerical rating given by the user (from 1 to 5).

## Steps and Analysis

### 1. Loading and Exploring the Data

We start by loading the dataset and checking its structure to understand the data we are working with.

### 2. Distribution of Ratings

We visualize the distribution of ratings to see how many reviews fall into each rating category (from 1 to 5). This gives us a clear picture of the overall sentiment towards LinkedIn as rated by users.

```python
sns.set(style="whitegrid")
plt.figure(figsize=(9, 5))
sns.countplot(data=linkedin_data, x='Rating')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
```
![Distribution of Ratings](https://github.com/aarish22/LinkedInReviewAnalysis/blob/main/Plots/Dostrobutionofratings.png)
### 3. Review Length Analysis

We calculate the length of each review to analyze the distribution of review lengths. This helps in understanding the level of detail provided by users in their reviews.

```python
linkedin_data['Review Length'] = linkedin_data['Review'].apply(len)

plt.figure(figsize=(9, 6))
sns.histplot(linkedin_data['Review Length'], bins=50, kde=True)
plt.title('Distribution of Review Lengths')
plt.xlabel('Length of Review')
plt.ylabel('Count')
plt.show()
```
![Distribution of Review Length](https://github.com/aarish22/LinkedInReviewAnalysis/blob/main/Plots/Distofreviewlengths.png)

### 4. Sentiment Analysis Using TextBlob

We perform sentiment analysis on the reviews using TextBlob. The sentiment is classified into three categories:

- **Positive**: Reviews with a polarity score greater than 0.1.
- **Negative**: Reviews with a polarity score less than -0.1.
- **Neutral**: Reviews with a polarity score between -0.1 and 0.1.

```python
from textblob import TextBlob

def textblob_sentiment_analysis(review):
    sentiment = TextBlob(review).sentiment
    if sentiment.polarity > 0.1:
        return 'Positive'
    elif sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

linkedin_data['Sentiment'] = linkedin_data['Review'].apply(textblob_sentiment_analysis)
```

### 5. Distribution of Sentiments

We analyze the distribution of sentiments across the dataset to see the overall sentiment breakdown.

```python
sentiment_distribution = linkedin_data['Sentiment'].value_counts()

plt.figure(figsize=(9, 5))
sns.barplot(x=sentiment_distribution.index, y=sentiment_distribution.values)
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
```
![Distribution of Sentiments](https://github.com/aarish22/LinkedInReviewAnalysis/blob/main/Plots/distofsentiments.png)

### 6. Sentiment Distribution Across Ratings

We further analyze how sentiments are distributed across different ratings to get insights into the correlation between the rating given and the sentiment expressed in the review.

```python
plt.figure(figsize=(10, 5))
sns.countplot(data=linkedin_data, x='Rating', hue='Sentiment')
plt.title('Sentiment Distribution Across Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.show()
```
![Sentiment Distribution Across Ratings](https://github.com/aarish22/LinkedInReviewAnalysis/blob/main/Plots/sentimentdistributionamonganalysis.png)

### 7. Word Cloud Generation

To visualize the most common words used in reviews with different sentiments, we generate word clouds for positive, negative, and neutral reviews.

```python
from wordcloud import WordCloud

def generate_word_cloud(sentiment):
    text = ' '.join(review for review in linkedin_data[linkedin_data['Sentiment'] == sentiment]['Review'])
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment} Reviews')
    plt.axis('off')
    plt.show()

for sentiment in ['Positive', 'Negative', 'Neutral']:
    generate_word_cloud(sentiment)
```
![Word Cloud](https://github.com/aarish22/LinkedInReviewAnalysis/blob/main/Plots/frequentwords.png)

## Conclusion

This project provides a comprehensive analysis of LinkedIn reviews by examining the distribution of ratings, the sentiment expressed in the reviews, and the key terms associated with each sentiment. This analysis can be helpful for understanding user feedback and improving LinkedIn's services.

## Dependencies

- pandas
- matplotlib
- seaborn
- numpy
- textblob
- wordcloud

Ensure you have these libraries installed before running the analysis.
