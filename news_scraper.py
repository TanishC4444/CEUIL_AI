import feedparser
from newspaper import Article
import os
import re
from time import sleep
from functools import lru_cache

# RSS feeds
FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "NPR": "https://feeds.npr.org/1001/rss.xml",
    "CNN": "http://rss.cnn.com/rss/cnn_topstories.rss",
    "PBS NewsHour": "https://www.pbs.org/newshour/feeds/rss/headlines",
    "Washington Post": "https://feeds.washingtonpost.com/rss/national",
    "NY Times": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "Texas Tribune": "https://feeds.texastribune.org/feeds/main/",
    "ABC Top Stories": "https://feeds.abcnews.com/abcnews/topstories",
    "ABC US Headlines": "https://feeds.abcnews.com/abcnews/usheadlines",
    "ABC Intl Headlines": "https://feeds.abcnews.com/abcnews/internationalheadlines",
    "ABC Politics": "https://feeds.abcnews.com/abcnews/politicsheadlines",
    "ABC Business": "https://feeds.abcnews.com/abcnews/businessheadlines",
    "ABC Tech": "https://feeds.abcnews.com/abcnews/technologyheadlines",
    "ABC Health": "https://feeds.abcnews.com/abcnews/healthheadlines",
    "ABC Entertainment": "https://feeds.abcnews.com/abcnews/entertainmentheadlines",    
    "CNBC Top News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", 
    "CNBC World News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362", 
    "CNBC US News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15837362", 
    "CNBC Asia News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19832390", 
    "CNBC EU News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19794221", 
    "CNBC Politics": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000113",
    "CNBC Business": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147",
    "CNBC Tech": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19854910",
    "CNBC Health": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000108",
    "CNBC Economy": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"
}

OUTPUT_DIR = "articles"
OUTPUT_FILE = "news_articles.txt"

# Compiled regex patterns for better performance
SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+')
NUMBER_PATTERN = re.compile(r'\d')
WHITESPACE_PATTERN = re.compile(r'\s+')

@lru_cache(maxsize=1000)
def has_number_or_proper_noun_cached(sentence):
    """Cached version of proper noun/number detection"""
    # Check for numbers
    if NUMBER_PATTERN.search(sentence):
        return True
    
    # Check for proper nouns (capitalized words that aren't at sentence start)
    words = sentence.split()
    for i, word in enumerate(words):
        # Skip first word (might be capitalized due to sentence start)
        if i > 0 and word and word[0].isupper() and word.isalpha():
            return True
    
    return False

def clean_text(text):
    """Optimized version of clean_text with faster regex and processing"""
    # Use compiled regex for better performance
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    number_pattern = re.compile(r'\d')
    
    sentences = sentence_pattern.split(text)
    
    filtered_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Quick length checks first (fastest operations)
        if len(sentence) < 50:  # Rough character estimate instead of word split
            continue
            
        # Check for repetition
        sentence_lower = sentence.lower()
        if sentence_lower in seen_sentences:
            continue
        seen_sentences.add(sentence_lower)
        
        # Only do expensive operations if needed
        words = sentence.split()
        if len(words) < 30:
            continue
            
        # Quick check for numbers
        has_number = number_pattern.search(sentence) is not None
        
        # Quick check for proper nouns (capitalized words not at start)
        has_proper_noun = any(word[0].isupper() and word.isalpha() 
                             for word in words[1:])
        
        if has_number or has_proper_noun:
            filtered_sentences.append(sentence)
    
    # Single join and regex clean
    cleaned = ' '.join(filtered_sentences)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def main():
    """Main function to scrape and process news articles"""
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    # Load existing URLs
    existing_urls = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Link: "):
                    url = line.replace("Link: ", "").strip()
                    existing_urls.add(url)

    print(f"Found {len(existing_urls)} existing articles")

    # Extract articles
    new_count = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for feed_name, feed_url in FEEDS.items():
            print(f"\nProcessing {feed_name}...")
            
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:20]:  # Limit per feed
                    article_url = entry.link
                    
                    if article_url in existing_urls:
                        continue
                    
                    try:
                        article = Article(article_url)
                        article.download()
                        article.parse()
                        
                        cleaned_text = clean_text(article.text)
                        
                        # Only keep if has content after filtering
                        if cleaned_text and len(cleaned_text.split()) > 30:
                            f.write(f"Link: {article_url}\n")
                            f.write(f"Article: {cleaned_text}\n\n")
                            
                            existing_urls.add(article_url)
                            new_count += 1
                            print(f"✅ Added: {entry.title[:50]}...")
                        else:
                            print(f"⏭️  Filtered out: {entry.title[:50]}...")
                            
                    except Exception as e:
                        print(f"❌ Failed: {entry.title[:50]}... - {e}")
                    
                    sleep(0.1)  # Be nice to servers
                    
            except Exception as e:
                print(f"❌ Failed to process {feed_name}: {e}")

    print(f"\n✅ Added {new_count} new articles to {output_path}")

if __name__ == "__main__":
    main()