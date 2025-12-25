#%%
from dotenv import load_dotenv
import os
from pathlib import Path
from newsapi import NewsApiClient
from time import sleep

# Get the path to the .env file in the root directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv('NEWSAPI_KEY')
print(f"API Key loaded: {api_key}")

newsapi = NewsApiClient(api_key=api_key)

#%%
# Fetch articles for all energy sources
energy_sources = ['solar energy', 'wind energy', 'hydro energy', 'nuclear energy', 'fossil fuels']
all_articles = {}

for source in energy_sources:
    try:
        articles = newsapi.get_everything(q=source, language='en', sort_by='publishedAt', page_size=10)
        all_articles[source] = articles['articles']
        print(f"Fetched {len(articles['articles'])} articles for {source}")
        sleep(1)  # Sleep for a second to avoid hitting the API rate limit
    except Exception as e:
        print(f"Error fetching articles for {source}: {e}")

#%%
# Print sample articles
for source, articles in all_articles.items():
    print(f"\n=== Articles for {source} ===")
    for i, article in enumerate(articles[:2]):  # Print first 2 articles for brevity
        print(f"\n--- Article {i+1} ---")
        print(f"Title: {article['title']}")
        print(f"Description: {article['description']}")
        print(f"Source: {article['source']['name']}")
        print(f"Published: {article['publishedAt']}")
        print(f"URL: {article['url']}")

#%%
# Initialize VADER
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk 

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

#%%
# Analyze solar articles with VADER
solar_articles = all_articles.get('solar energy', [])

for article in solar_articles:
    title = article['title']
    description = article['description']
    
    text_to_analyze = f"{title}. {description}" if description else title

    if text_to_analyze:
        score = sia.polarity_scores(text_to_analyze)
        author = article['author']
        date = article['publishedAt']
        
        print(f"Title: {title}")
        print(f"Description: {description}")
        print(f"Sentiment Score: {score}")
        print(f"Compound: {score['compound']:.3f}")
        print("\n")
# %%
