import feedparser
from newspaper import Article
import os
import re
from time import sleep
from llama_cpp import Llama
import concurrent.futures
import threading
from functools import lru_cache

# RSS feeds
FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "NPR": "https://feeds.npr.org/1001/rss.xml",
    "CNN": "http://rss.cnn.com/rss/cnn_topstories.rss",
    "PBS NewsHour": "https://www.pbs.org/newshour/feeds/rss/headlines",
    "Washington Post": "https://feeds.washingtonpost.com/rss/national",
    "NY Times": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"
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
                    print(f"✅ Added: {entry.title[:30]}...")
                else:
                    print(f"⏭️  Filtered out: {entry.title[:30]}...")
                    
            except Exception as e:
                print(f"❌ Failed: {entry.title[:30]}... - {e}")
            
            sleep(0.1)  # Be nice to servers

print(f"\n✅ Added {new_count} new articles to {output_path}")

def generate_mcqs(article_text):
    """Generate MCQs from article text"""
    # Only use first chunk if article is very long to avoid repetition
    chunks = chunk_text(article_text)
    if len(chunks) > 1:
        # Use only the most substantial chunk to avoid model confusion
        chunks = [max(chunks, key=len)]
    
    all_questions = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk.split())} words)...")
        
        # Skip if chunk is too short
        if len(chunk.split()) < 100:
            print("Chunk too short, skipping...")
            continue
            
        prompt = PROMPT_TEMPLATE.format(article=chunk)
        
        try:
            response = llm(
                prompt,
                max_tokens=600,  # Reduced to prevent hallucination
                temperature=0.3,  # Lower temperature for more focused output
                stop=["Article:", "\n\nHere", "Instructions:", "Format your"]
            )
            output = response['choices'][0]['text'].strip()
            
            # More aggressive cleaning
            lines = output.split('\n')
            cleaned_lines = []
            question_count = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Stop if we see repetitive content
                if line.count('"') > 10 or len(line) > 200:
                    break
                    
                # Count questions to limit to 3
                if line.startswith('Q') and line[1:2].isdigit():
                    question_count += 1
                    if question_count > 3:
                        break
                        
                # Skip problematic lines
                if any(skip_phrase in line.lower() for skip_phrase in 
                      ['here is', 'write 3', 'instructions', 'format your', 'article:']):
                    break
                    
                cleaned_lines.append(line)
            
            cleaned_output = '\n'.join(cleaned_lines)
            
            # Validate output has at least one complete question
            if 'Q1.' in cleaned_output and 'Correct Answer:' in cleaned_output:
                all_questions.append(cleaned_output)
            else:
                print("Generated output doesn't contain valid questions, skipping...")
                
        except Exception as e:
            print(f"Error generating MCQs for chunk {i+1}: {e}")
            continue
    
    return '\n\n'.join(all_questions)

# Initialize your LLaMA model with optimized settings
# Use environment variable for model path, fallback to local path
MODEL_PATH = os.getenv('MODEL_PATH', '/Users/tanishchauhan/Desktop/CEUIL_AI/mistral-7b-instruct-v0.1.Q4_K_M.gguf')

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,  # Reduced context window for faster processing
    n_gpu_layers=-1,  # Use all GPU layers if available
    n_threads=8,  # Adjust based on your CPU cores
    n_batch=512,  # Larger batch size for better GPU utilization
    use_mlock=True,  # Keep model in memory
    use_mmap=True,  # Memory-mapped file access
    verbose=False  # Reduce logging overhead
)

# Improved prompt template with stricter formatting
PROMPT_TEMPLATE = """Create exactly 5 multiple choice questions from this news article for UIL Current Events competition.

STRICT REQUIREMENTS:
- Base questions ONLY on facts explicitly stated in the article
- Each question must have exactly 4 options (A, B, C, D)
- Each question must have exactly 1 correct answer (A, B, C, or D - just the letter)
- Focus on specific names, numbers, locations, dates mentioned in the article
- No trick questions or "all of the above" type answers

FORMAT (follow exactly):
Q1. [Specific factual question about the article]
A. [Option A]
B. [Option B] 
C. [Option C]
D. [Option D]
Correct Answer: [Single letter: A, B, C, or D]

Q2. [Specific factual question about the article]
A. [Option A]
B. [Option B]
C. [Option C] 
D. [Option D]
Correct Answer: [Single letter: A, B, C, or D]

Q3. [Specific factual question about the article]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
Correct Answer: [Single letter: A, B, C, or D]

Article: {article}

Generate exactly 3 questions now:
"""

def read_articles(filename):
    """Fixed function to properly parse the article file"""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    articles = []
    # Split by double newline to separate articles
    article_blocks = content.split('\n\n')
    
    current_link = None
    current_article = None
    
    for block in article_blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        
        for line in lines:
            if line.startswith("Link: "):
                current_link = line.strip()
            elif line.startswith("Article: "):
                current_article = line[len("Article: "):].strip()
                
                # If we have both link and article, add to list
                if current_link and current_article:
                    articles.append((current_link, current_article))
                    current_link = None
                    current_article = None
    
    return articles

def chunk_text(text, max_words=600):
    """Chunk text by words, ensuring we don't split sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed limit, start new chunk
        if current_word_count + sentence_words > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words

    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def extract_headline_from_url(url):
    """Extract a readable headline from the URL or article content"""
    try:
        # Try to get headline from URL structure
        if '/articles/' in url:
            # BBC style URLs
            article_id = url.split('/articles/')[-1].split('?')[0]
            return f"BBC News Article ({article_id})"
        elif '/news/' in url:
            # Other news URLs
            parts = url.split('/')
            if len(parts) > 4:
                return f"News Article: {parts[-1].replace('-', ' ').replace('.html', '').title()}"
    except:
        pass
    
    # Fallback to domain name
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return f"News Article from {domain}"
    except:
        return "News Article"

def generate_quick_summary_with_location(article_text):
    """Generate a brief summary mentioning location but avoiding other specific details"""
    # Use first 300 words for summary
    words = article_text.split()
    summary_text = ' '.join(words[:300]) if len(words) > 300 else article_text
    
    summary_prompt = f"""Write a 1-2 sentence summary of this news article. Include the main location/country involved but avoid specific names of people, organizations, exact numbers, or detailed facts that could be quiz answers.

Format: "This article reports on [type of event] in [location/country] involving [general description]."

Article excerpt: {summary_text}

Summary:"""

    try:
        response = llm(
            summary_prompt,
            max_tokens=60,  # Even shorter for speed
            temperature=0.2,
            stop=["\n", "Article:", "Summary:"]
        )
        summary = response['choices'][0]['text'].strip()
        
        if summary and len(summary) > 15:
            return summary
        else:
            return "This article covers recent international news developments."
            
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "This article covers recent news developments."

def generate_mcqs_optimized(article_text):
    """Optimized MCQ generation with reduced processing"""
    # Skip chunking for shorter articles - process directly
    word_count = len(article_text.split())
    
    if word_count > 800:
        # Only chunk if really necessary, use larger chunks
        chunks = chunk_text(article_text, max_words=800)
        chunk = chunks[0]  # Just use first chunk
    else:
        chunk = article_text
    
    print(f"Processing single chunk ({len(chunk.split())} words)...")
    
    if len(chunk.split()) < 100:
        return ""
    
    prompt = PROMPT_TEMPLATE.format(article=chunk)
    
    try:
        response = llm(
            prompt,
            max_tokens=500,  # Reduced for faster generation
            temperature=0.2,  # Lower for consistency and speed
            top_p=0.9,  # Add top_p for better control
            stop=["Article:", "\n\nHere", "Instructions:"],
            echo=False  # Don't echo input
        )
        
        output = response['choices'][0]['text'].strip()
        
        # Faster validation - just check for basic structure
        if 'Q1.' in output and 'Q2.' in output and 'Q3.' in output:
            # Simple cleanup - remove anything after Q3's answer
            lines = output.split('\n')
            cleaned_lines = []
            q3_answer_found = False
            
            for line in lines:
                cleaned_lines.append(line)
                if q3_answer_found and line.strip().startswith('Q'):
                    break
                if line.startswith('Correct Answer:') and 'Q3.' in '\n'.join(cleaned_lines[-10:]):
                    q3_answer_found = True
                    break
            
            return '\n'.join(cleaned_lines)
        else:
            print("Generated output missing required questions")
            return ""
            
    except Exception as e:
        print(f"Error generating MCQs: {e}")
        return ""

def main():
    # Use environment variable for input file path, fallback to local path
    input_file = os.getenv('INPUT_FILE', '/Users/tanishchauhan/Desktop/CEUIL_AI/articles/news_articles.txt')
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found!")
        return
    
    articles = read_articles(input_file)
    print(f"Found {len(articles)} articles to process")
    
    if not articles:
        print("No articles found in the input file!")
        return
    
    USE_HEADLINE_ONLY = True
    
    with open('quiz.txt', 'w', encoding='utf-8') as out:
        for i, (link, article) in enumerate(articles):
            if i >= 1:  # Process only first 3 articles to avoid overwhelming output
                break
                
            print(f"\nProcessing article {i+1}: {link}")
            print(f"Original article length: {len(article.split())} words")
            
            # Clean the article text more aggressively
            cleaned_article = clean_text(article)
            print(f"Cleaned article length: {len(cleaned_article.split())} words")
            
            if len(cleaned_article.split()) < 100:
                print("Article too short after cleaning, skipping...")
                continue

            # Generate summary or extract headline
            if USE_HEADLINE_ONLY:
                article_info = extract_headline_from_url(link)
                print("Using headline extraction (fast mode)")
            else:
                print("Generating article summary with location...")
                article_info = generate_quick_summary_with_location(cleaned_article)

            
            out.write(f"{link}\n")
            out.write(f"Info: {' '.join((article.split())[:15])}\n")
            out.write("="*80 + "\n")
            
            mcqs = generate_mcqs_optimized(cleaned_article)
            if mcqs.strip():
                out.write(mcqs + '\n\n')
            else:
                out.write("No valid questions could be generated for this article.\n\n")
            out.write("="*80 + "\n\n")
    
    print(f"\n✅ MCQs generated and saved to quiz.txt")

if __name__ == "__main__":
    main()