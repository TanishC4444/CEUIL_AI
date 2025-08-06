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
PROMPT_TEMPLATE = """Create exactly 5 multiple choice questions from this news article for UIL Current Events competition.

STRICT REQUIREMENTS:
- Base questions ONLY on facts explicitly stated in the article
- Each question must be clear and unambiguous
- Each question must have exactly 4 options (A, B, C, D)
- Each question must have exactly 1 correct answer (A, B, C, or D - just the letter)
- Focus on specific names, numbers, locations, dates mentioned in the article
- No trick questions or "all of the above" type answers
- Generate questions that test factual recall, not inference or opinion
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
    """Read articles from file and return list of (link, article) tuples"""
    if not os.path.exists(filename):
        return []
        
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

def remove_processed_articles(filename, processed_links):
    """Remove processed articles from the input file - FIXED VERSION"""
    if not os.path.exists(filename) or not processed_links:
        return
    
    print(f"Removing {len(processed_links)} processed articles from {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if not content.strip():
        return
    
    # Split into article blocks
    article_blocks = content.split('\n\n')
    remaining_blocks = []
    
    for block in article_blocks:
        if not block.strip():
            continue
            
        # Check if this block contains a processed link
        block_should_be_kept = True
        
        for processed_link in processed_links:
            if processed_link in block:
                block_should_be_kept = False
                break
        
        if block_should_be_kept:
            remaining_blocks.append(block)
    
    # Write remaining articles back to file
    with open(filename, 'w', encoding='utf-8') as f:
        if remaining_blocks:
            f.write('\n\n'.join(remaining_blocks))
            f.write('\n\n')  # Add final newlines
        # If no remaining blocks, file will be empty (which is correct)
    
    print(f"✅ Kept {len(remaining_blocks)} articles, removed {len(processed_links)} articles")

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
    """Extract a readable headline from the URL"""
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
        domain = urlparse(url).netlnet
        return f"News Article from {domain}"
    except:
        return "News Article"

def generate_mcqs_optimized(article_text):
    """Generate MCQs from article text with timeout protection"""
    # Skip chunking for shorter articles - process directly
    word_count = len(article_text.split())
    
    if word_count > 800:
        # Only chunk if really necessary, use larger chunks
        chunks = chunk_text(article_text, max_words=800)
        chunk = chunks[0]  # Just use first chunk
    else:
        chunk = article_text
    
    print(f"Processing chunk ({len(chunk.split())} words)...")
    
    if len(chunk.split()) < 100:
        print("Chunk too short, skipping...")
        return ""
    
    prompt = PROMPT_TEMPLATE.format(article=chunk)
    
    try:
        start_time = time.time()
        
        response = llm(
            prompt,
            max_tokens=400,  # Reduced further for speed
            temperature=0.1,  # Even lower for faster generation
            top_p=0.9,
            stop=["Article:", "\n\nHere", "Instructions:"],
            echo=False
        )
        
        generation_time = time.time() - start_time
        print(f"Generated in {generation_time:.1f}s")
        
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
    """Main function to process articles and generate MCQs - OPTIMIZED FOR GITHUB ACTIONS"""
    input_file = os.getenv('INPUT_FILE', '/Users/tanishchauhan/Desktop/CEUIL_AI/articles/news_articles.txt')
    
    articles = read_articles(input_file)
    print(f"Found {len(articles)} articles to process")
    
    if not articles:
        print("No articles found in the input file!")
        return
    
    # PROCESS ALL ARTICLES (or large batches) - let GitHub Actions 6-hour limit handle it
    batch_size = min(500, len(articles))  # Process up to 500 articles per run
    articles_to_process = articles[:batch_size]
    
    print(f"Processing {len(articles_to_process)} articles in this batch...")
    
    processed_links = []
    successful_count = 0
    start_time = time.time()
    
    # Open quiz file in append mode to add to existing content
    with open('quiz.txt', 'a', encoding='utf-8') as out:
        for i, (link, article) in enumerate(articles_to_process):
            print(f"\n--- Processing article {i+1}/{len(articles_to_process)} ---")
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1) if i > 0 else 0
            est_remaining = avg_time * (len(articles_to_process) - i - 1)
            
            print(f"Link: {link}")
            print(f"Article length: {len(article.split())} words")
            print(f"Elapsed: {elapsed/60:.1f}min, Avg: {avg_time:.1f}s/article")
            
            # Skip articles that are too short
            if len(article.split()) < 100:
                print("Article too short after cleaning, skipping...")
                processed_links.append(link)
                continue

            # Extract headline from URL
            article_info = extract_headline_from_url(link)
            
            # Generate MCQs with progress indication
            mcqs = generate_mcqs_optimized(article)
            
            # Write to output file
            out.write(f"\n{link}\n")
            out.write(f"Info: {article_info}\n")
            out.write("="*80 + "\n")
            
            if mcqs.strip():
                out.write(mcqs + '\n\n')
                successful_count += 1
                print("✅ MCQs generated successfully")
            else:
                out.write("No valid questions could be generated for this article.\n\n")
                print("❌ Failed to generate valid MCQs")
                
            out.write("="*80 + "\n\n")
            
            # Mark as processed regardless of success/failure
            processed_links.append(link)
    
    total_time = time.time() - start_time
    print(f"\n✅ Batch complete: {successful_count}/{len(processed_links)} articles processed in {total_time/60:.1f} minutes")
    
    # Remove processed articles from input file - THIS WAS THE BUG
    if processed_links:
        print(f"Removing {len(processed_links)} processed articles from input file...")
        remove_processed_articles(input_file, processed_links)
        print("✅ Input file updated")
    
    # Show remaining work
    remaining_articles = read_articles(input_file)
    print(f"📊 Remaining articles to process: {len(remaining_articles)}")

if __name__ == "__main__":
    main()