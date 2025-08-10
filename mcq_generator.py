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

def read_articles(filename):
    """Read articles from file and return list of (link, article) tuples with improved parsing"""
    if not os.path.exists(filename):
        print(f"Input file {filename} does not exist!")
        return []
        
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return []

    if not content.strip():
        print(f"Input file {filename} is empty!")
        return []

    articles = []
    # Normalize line endings first
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    
    # Split by double newline, but be more flexible
    article_blocks = re.split(r'\n\s*\n+', content.strip())
    
    for block in article_blocks:
        block = block.strip()
        if not block:
            continue
            
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        current_link = None
        article_lines = []
        in_article = False
        
        for line in lines:
            if line.startswith("Link: "):
                current_link = line
                in_article = False
                article_lines = []
            elif line.startswith("Article: "):
                # Start collecting article text
                article_content = line[len("Article: "):].strip()
                if article_content:  # If there's content on the same line
                    article_lines = [article_content]
                else:
                    article_lines = []
                in_article = True
            elif in_article and current_link:
                # Continue collecting article text
                article_lines.append(line)
        
        # Join all article lines and add to articles list
        if current_link and article_lines:
            current_article = ' '.join(article_lines).strip()
            if current_article and len(current_article) > 50:  # Only add substantial articles
                articles.append((current_link, current_article))
                print(f"✓ Parsed article: {current_link[:60]}...")
    
    print(f"Successfully parsed {len(articles)} articles from {filename}")
    return articles

def write_remaining_articles(filename, remaining_articles):
    """Write remaining articles back to the input file with better error handling"""
    try:
        # Create backup first
        backup_filename = filename + '.backup'
        if os.path.exists(filename):
            import shutil
            shutil.copy2(filename, backup_filename)
            print(f"✓ Created backup: {backup_filename}")
        
        # Write the remaining articles
        with open(filename, 'w', encoding='utf-8') as f:
            if remaining_articles:
                for i, (link, article) in enumerate(remaining_articles):
                    f.write(f"{link}\n")
                    f.write(f"Article: {article}\n")
                    if i < len(remaining_articles) - 1:  # Add separator between articles
                        f.write("\n")
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force write to disk
        
        # Verify the write was successful
        if remaining_articles:
            verification = read_articles(filename)
            if len(verification) != len(remaining_articles):
                print(f"⚠️ Warning: Expected {len(remaining_articles)} articles but found {len(verification)} after write")
                # Restore from backup if verification fails
                if os.path.exists(backup_filename):
                    import shutil
                    shutil.copy2(backup_filename, filename)
                    print(f"❌ Write verification failed, restored from backup")
                    return False
            else:
                print(f"✅ Write verification successful: {len(remaining_articles)} articles")
        else:
            print(f"✅ Successfully cleared input file - no articles remaining")
        
        # Clean up backup after successful write
        if os.path.exists(backup_filename):
            os.remove(backup_filename)
        
        return True
        
    except Exception as e:
        print(f"❌ Error writing to input file: {e}")
        # Try to restore from backup
        backup_filename = filename + '.backup'
        if os.path.exists(backup_filename):
            try:
                import shutil
                shutil.copy2(backup_filename, filename)
                print(f"✅ Restored from backup after write error")
            except:
                print(f"❌ Failed to restore from backup")
        return False

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
            max_tokens=500,  # Reduced further for speed
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
    """Main function to process articles and generate MCQs - FIXED VERSION"""
    input_file = os.getenv('INPUT_FILE', '/Users/tanishchauhan/Desktop/CEUIL_AI/articles/news_articles.txt')
    
    print(f"🔍 Reading articles from: {input_file}")
    print(f"📁 Input file exists: {os.path.exists(input_file)}")
    if os.path.exists(input_file):
        print(f"📄 Input file size: {os.path.getsize(input_file)} bytes")
    
    # Read all articles at start
    all_articles = read_articles(input_file)
    print(f"Found {len(all_articles)} total articles to process")
    
    if not all_articles:
        print("No articles found in the input file!")
        return
    
    # PROCESS ARTICLES IN BATCHES - FIXED BATCH PROCESSING
    batch_size = min(450, len(all_articles))  # Process up to 450 articles per run
    articles_to_process = all_articles[:batch_size]
    remaining_articles = all_articles[batch_size:]  # Articles to keep for next run
    
    print(f"📋 Processing {len(articles_to_process)} articles in this batch...")
    print(f"📋 Will keep {len(remaining_articles)} articles for next run...")
    
    # Show first few links being processed for debugging
    print("\n🔍 First few articles being processed:")
    for i, (link, _) in enumerate(articles_to_process[:3]):
        print(f"  {i+1}. {link}")
    
    if remaining_articles:
        print("\n🔍 First few articles remaining for next run:")
        for i, (link, _) in enumerate(remaining_articles[:3]):
            print(f"  {i+1}. {link}")
    
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
            print(f"Est. remaining: {est_remaining/60:.1f}min")

            # Skip articles that are too short
            if len(article.split()) < 100:
                print("Article too short after cleaning, skipping...")
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
            out.flush()  # Ensure quiz content is written immediately
    
    total_time = time.time() - start_time
    print(f"\n✅ Batch complete: {successful_count}/{len(articles_to_process)} articles processed in {total_time/60:.1f} minutes")
    
    # CRITICAL FIX: Write remaining articles back with verification
    print(f"\n📝 Updating input file:")
    print(f"   - Original articles: {len(all_articles)}")
    print(f"   - Processed articles: {len(articles_to_process)}")
    print(f"   - Remaining articles: {len(remaining_articles)}")
    
    # Attempt to write remaining articles with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        print(f"\n📝 Write attempt {attempt + 1}/{max_retries}")
        
        if write_remaining_articles(input_file, remaining_articles):
            print("✅ Successfully updated input file")
            break
        else:
            if attempt < max_retries - 1:
                print("⚠️ Write failed, retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("❌ Failed to update input file after all attempts")
                # Create a fallback file
                fallback_file = input_file + '.remaining'
                try:
                    write_remaining_articles(fallback_file, remaining_articles)
                    print(f"💾 Created fallback file: {fallback_file}")
                except:
                    print("❌ Even fallback write failed!")
                return
    
    # Verify the write operation worked
    print(f"\n🔍 Verifying file update...")
    verification_articles = read_articles(input_file)
    expected_count = len(remaining_articles)
    actual_count = len(verification_articles)
    
    print(f"✅ Verification complete:")
    print(f"   - Expected articles: {expected_count}")
    print(f"   - Actual articles: {actual_count}")
    print(f"   - Match: {'✅ YES' if expected_count == actual_count else '❌ NO'}")
    
    # Show final status
    print(f"\n📊 FINAL STATUS:")
    print(f"📊 Articles processed this run: {len(articles_to_process)}")
    print(f"📊 Articles successfully generated MCQs: {successful_count}")
    print(f"📊 Remaining articles for next run: {actual_count}")
    
    if actual_count > 0:
        print("💡 Next run will process more articles")
    else:
        print("🎉 All articles have been processed!")

if __name__ == "__main__":
    main()