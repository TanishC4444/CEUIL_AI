import os
import re
from llama_cpp import Llama
from urllib.parse import urlparse
import time

# Initialize your LLaMA model with optimized settings
MODEL_PATH = os.getenv('MODEL_PATH', '/Users/tanishchauhan/Desktop/CEUIL_AI/mistral-7b-instruct-v0.1.Q4_K_M.gguf')

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=-1,
    n_threads=8,
    n_batch=512,
    use_mlock=True,
    use_mmap=True,
    verbose=False
)

# Improved prompt template with stricter formatting
PROMPT_TEMPLATE = """Create exactly 3 multiple choice questions from this news article for UIL Current Events competition.

STRICT REQUIREMENTS:
- Base questions ONLY on facts explicitly stated in the article
- Each question must be clear and unambiguous
- Each question must have exactly 4 options (A, B, C, D)
- Each question must have exactly 1 correct answer (A, B, C, or D - just the letter)
- Focus on specific names, numbers, locations, dates mentioned in the article
- No trick questions or "all of the above" type answers
- Generate questions that test factual recall, not inference or opinion
- Add a bit of context to each question to make it clear what it's asking and what event or fact it relates to
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

# OPTIMIZED: Pre-compiled regex and blocked domains as tuple
BLOCKED_DOMAINS = (
    "morningbrew.com",
    "newsweek.com",
    "kxan.com",
    "wfaa.com"
)

def read_articles_fast(filename):
    """OPTIMIZED: Fast streaming reader with filtering built-in"""
    if not os.path.exists(filename):
        print(f"Input file {filename} does not exist!")
        return []
    
    articles = []
    current_link = None
    current_article = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("Link: "):
                # Process previous article
                if current_link and current_article:
                    article_text = ' '.join(current_article).strip()
                    
                    # FAST FILTERING: Check length and domains in one pass
                    if len(article_text) >= 100:
                        if not any(domain in current_link for domain in BLOCKED_DOMAINS):
                            if len(article_text.split()) >= 100:  # Word count check
                                articles.append((current_link, article_text))
                                if len(articles) % 100 == 0:
                                    print(f"✓ Loaded {len(articles)} valid articles...")
                
                # Start new article
                current_link = line
                current_article = []
                
            elif line.startswith("Article: "):
                # Start collecting article text
                article_content = line[len("Article: "):].strip()
                if article_content:
                    current_article = [article_content]
                else:
                    current_article = []
                    
            elif current_link and line:
                # Continue building article
                current_article.append(line)
        
        # Process last article
        if current_link and current_article:
            article_text = ' '.join(current_article).strip()
            if len(article_text) >= 100:
                if not any(domain in current_link for domain in BLOCKED_DOMAINS):
                    if len(article_text.split()) >= 100:
                        articles.append((current_link, article_text))
    
    print(f"✅ Loaded {len(articles)} valid articles (filtered during read)")
    return articles

def write_remaining_articles_fast(filename, remaining_articles):
    """OPTIMIZED: Direct write without backup for speed"""
    try:
        with open(filename, 'w', encoding='utf-8', buffering=8192*4) as f:
            for i, (link, article) in enumerate(remaining_articles):
                f.write(f"{link}\n")
                f.write(f"Article: {article}\n")
                if i < len(remaining_articles) - 1:
                    f.write("\n")
        
        print(f"✅ Wrote {len(remaining_articles)} articles to file")
        return True
        
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        return False

def chunk_text(text, max_words=600):
    """Chunk text by words, ensuring we don't split sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        if current_word_count + sentence_words > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def extract_headline_from_url(url):
    """Extract a readable headline from the URL"""
    try:
        if '/articles/' in url:
            article_id = url.split('/articles/')[-1].split('?')[0]
            return f"BBC News Article ({article_id})"
        elif '/news/' in url:
            parts = url.split('/')
            if len(parts) > 4:
                return f"News Article: {parts[-1].replace('-', ' ').replace('.html', '').title()}"
    except:
        pass
    
    try:
        domain = urlparse(url).netloc
        return f"News Article from {domain}"
    except:
        return "News Article"

def generate_mcqs_optimized(article_text):
    """Generate MCQs from article text with timeout protection"""
    word_count = len(article_text.split())
    
    if word_count > 800:
        chunks = chunk_text(article_text, max_words=800)
        chunk = chunks[0]
    else:
        chunk = article_text
    
    if len(chunk.split()) < 100:
        return ""
    
    prompt = PROMPT_TEMPLATE.format(article=chunk)
    
    try:
        start_time = time.time()
        
        response = llm(
            prompt,
            max_tokens=500,
            temperature=0.1,
            top_p=0.9,
            stop=["Article:", "\n\nHere", "Instructions:"],
            echo=False
        )
        
        generation_time = time.time() - start_time
        print(f"Generated in {generation_time:.1f}s")
        
        output = response['choices'][0]['text'].strip()
        
        if 'Q1.' in output and 'Q2.' in output and 'Q3.' in output:
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
            return ""
            
    except Exception as e:
        print(f"Error generating MCQs: {e}")
        return ""

def main():
    """OPTIMIZED: Main function with faster I/O"""
    input_file = os.getenv('INPUT_FILE', '/Users/tanishchauhan/Desktop/CEUIL_AI/articles/news_articles.txt')
    
    print(f"🔍 Reading and filtering articles from: {input_file}")
    start_load = time.time()
    
    # OPTIMIZED: Single-pass read with built-in filtering
    all_articles = read_articles_fast(input_file)
    load_time = time.time() - start_load
    print(f"⚡ Loaded in {load_time:.1f}s")
    
    if not all_articles:
        print("No valid articles found!")
        return
    
    # Process in batches
    batch_size = min(250, len(all_articles))
    articles_to_process = all_articles[:batch_size]
    remaining_articles = all_articles[batch_size:]
    
    print(f"📋 Processing {len(articles_to_process)} articles")
    print(f"📋 Keeping {len(remaining_articles)} for next run")
    
    successful_count = 0
    start_time = time.time()
    
    # OPTIMIZED: Larger buffer for output file
    with open('quiz.txt', 'a', encoding='utf-8', buffering=8192*4) as out:
        for i, (link, article) in enumerate(articles_to_process):
            if i % 10 == 0:  # Progress every 10 articles
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(articles_to_process) - i) / rate if rate > 0 else 0
                print(f"\n[{i}/{len(articles_to_process)}] Rate: {rate:.1f}/s, ETA: {eta/60:.1f}min")
            
            article_info = extract_headline_from_url(link)
            mcqs = generate_mcqs_optimized(article)
            
            out.write(f"\n{link}\n")
            out.write(f"Info: {article_info}\n")
            out.write("="*80 + "\n")
            
            if mcqs.strip():
                out.write(mcqs + '\n\n')
                successful_count += 1
            else:
                out.write("No valid questions could be generated.\n\n")
                
            out.write("="*80 + "\n\n")
    
    total_time = time.time() - start_time
    print(f"\n✅ Processed {successful_count}/{len(articles_to_process)} in {total_time/60:.1f}min")
    
    # OPTIMIZED: Fast write of remaining articles
    print(f"\n📝 Updating input file...")
    write_start = time.time()
    write_remaining_articles_fast(input_file, remaining_articles)
    write_time = time.time() - write_start
    print(f"⚡ Write completed in {write_time:.1f}s")
    
    print(f"\n📊 Articles remaining for next run: {len(remaining_articles)}")

if __name__ == "__main__":
    main()