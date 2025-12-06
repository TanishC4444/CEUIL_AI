import os
import re
from llama_cpp import Llama
from urllib.parse import urlparse
import time

# Initialize TinyLlama model with optimized settings
MODEL_PATH = os.getenv('MODEL_PATH', '/Users/tanishchauhan/Desktop/CEUIL_AI/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')

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

# ENHANCED prompt template with stricter formatting and examples
PROMPT_TEMPLATE = """<|system|>
You are a UIL Current Events expert. Your ONLY job is to create exactly 3 multiple choice questions from news articles. Follow the format EXACTLY as shown.</|system|>
<|user|>
Read this article carefully and create EXACTLY 3 multiple choice questions.

CRITICAL RULES - DO NOT BREAK THESE:
1. Create EXACTLY 3 questions (Q1, Q2, Q3) - no more, no less
2. Each question MUST have EXACTLY 4 answer choices labeled A, B, C, D
3. Each question MUST end with "Correct Answer: [single letter]"
4. Base questions ONLY on facts explicitly stated in the article
5. Use specific names, numbers, dates, and locations from the article
6. Make wrong answers plausible but clearly incorrect
7. Do NOT add any extra text, explanations, or commentary
8. Do NOT create questions about things not mentioned in the article

EXACT FORMAT TO FOLLOW (copy this structure):

Q1. [Ask about a specific person, place, number, or date from the article]
A. [Wrong answer - plausible but incorrect]
B. [Correct answer - exact fact from article]
C. [Wrong answer - plausible but incorrect]
D. [Wrong answer - plausible but incorrect]
Correct Answer: B

Q2. [Ask about a different specific fact from the article]
A. [Correct answer - exact fact from article]
B. [Wrong answer - plausible but incorrect]
C. [Wrong answer - plausible but incorrect]
D. [Wrong answer - plausible but incorrect]
Correct Answer: A

Q3. [Ask about another specific fact from the article]
A. [Wrong answer - plausible but incorrect]
B. [Wrong answer - plausible but incorrect]
C. [Wrong answer - plausible but incorrect]
D. [Correct answer - exact fact from article]
Correct Answer: D

EXAMPLE (this is what good questions look like):

Article: "President Smith announced a $50 million aid package for disaster relief in Florida on March 15, 2024. The funds will support rebuilding efforts in Miami."

Q1. How much money did President Smith announce for disaster relief?
A. $25 million
B. $50 million
C. $75 million
D. $100 million
Correct Answer: B

Q2. Which state will receive the disaster relief funds?
A. Florida
B. Texas
C. California
D. Louisiana
Correct Answer: A

Q3. When was the aid package announced?
A. March 10, 2024
B. March 12, 2024
C. March 15, 2024
D. March 20, 2024
Correct Answer: C

NOW CREATE YOUR QUESTIONS FROM THIS ARTICLE:

Article: {article}

REMEMBER: EXACTLY 3 questions (Q1, Q2, Q3), each with 4 options (A, B, C, D), and each ending with "Correct Answer: [letter]"

BEGIN YOUR QUESTIONS NOW:</|user|>
<|assistant|>
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

def chunk_text(text, max_words=500):
    """Chunk text by words, ensuring we don't split sentences - smaller for TinyLlama"""
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
    """Generate MCQs from article text using TinyLlama with enhanced prompt"""
    word_count = len(article_text.split())
    
    # TinyLlama works better with shorter context
    if word_count > 600:
        chunks = chunk_text(article_text, max_words=600)
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
            max_tokens=700,  # More tokens to ensure complete output
            temperature=0.2,  # Slightly higher for creativity but still focused
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.15,  # Stronger penalty to avoid repetition
            stop=["</|assistant|>", "<|user|>", "<|system|>", "Article:", "\n\nNOW CREATE", "\n\nREMEMBER:"],
            echo=False
        )
        
        generation_time = time.time() - start_time
        print(f"Generated in {generation_time:.1f}s", end=" ")
        
        output = response['choices'][0]['text'].strip()
        
        # Enhanced validation
        if 'Q1.' in output and 'Q2.' in output and 'Q3.' in output:
            # More robust extraction
            lines = output.split('\n')
            cleaned_lines = []
            q_count = 0
            answer_count = 0
            
            for line in lines:
                line = line.strip()
                
                # Stop if we see example text bleeding through
                if "EXAMPLE" in line or "this is what good questions" in line.lower():
                    break
                
                # Count questions
                if line.startswith('Q1.') or line.startswith('Q2.') or line.startswith('Q3.'):
                    q_count += 1
                
                # Count answers
                if line.startswith('Correct Answer:'):
                    answer_count += 1
                    cleaned_lines.append(line)
                    # Stop after Q3's answer
                    if answer_count >= 3:
                        break
                    continue
                
                # Only add valid lines
                if line and (line[0] in 'QABCD' or line.startswith('Correct')):
                    cleaned_lines.append(line)
            
            # Validate we have complete questions
            result = '\n'.join(cleaned_lines)
            if q_count >= 3 and answer_count >= 3:
                print("✓")
                return result
            else:
                print(f"✗ (Q:{q_count}, A:{answer_count})")
                return ""
        else:
            print("✗ (missing questions)")
            return ""
            
    except Exception as e:
        print(f"✗ Error: {e}")
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
    
    # Process in batches - TinyLlama is very fast
    batch_size = min(400, len(all_articles))  # Increased for speed
    articles_to_process = all_articles[:batch_size]
    remaining_articles = all_articles[batch_size:]
    
    print(f"📋 Processing {len(articles_to_process)} articles with TinyLlama")
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
            print(f"  Processing: {article_info[:50]}... ", end="")
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
    avg_time = total_time / len(articles_to_process) if articles_to_process else 0
    print(f"\n✅ Processed {successful_count}/{len(articles_to_process)} in {total_time/60:.1f}min")
    print(f"⚡ Average: {avg_time:.1f}s per article")
    print(f"📊 Success rate: {successful_count/len(articles_to_process)*100:.1f}%")
    
    # OPTIMIZED: Fast write of remaining articles
    print(f"\n📝 Updating input file...")
    write_start = time.time()
    write_remaining_articles_fast(input_file, remaining_articles)
    write_time = time.time() - write_start
    print(f"⚡ Write completed in {write_time:.1f}s")
    
    print(f"\n📊 Articles remaining for next run: {len(remaining_articles)}")

if __name__ == "__main__":
    main()