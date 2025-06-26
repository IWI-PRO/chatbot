from scraper import scrape_website
from chunking import chunk_text
from embedder import TextEmbedder
from faiss_store import FAISSStore
from ollama_query import query_ollama

def main():
    # Step 1: Scrape
    base_url = "https://www.makautexam.net/"
    print(f"🌐 Scraping from: {base_url}")
    scraped_pages = scrape_website(base_url)
    if not scraped_pages:
        print("❌ No text scraped. Exiting.")
        return

    # Step 2: Chunk
    chunks = []
    for page_text in scraped_pages:
        chunks.extend(chunk_text(page_text))
    if not chunks:
        print("❌ No chunks created. Exiting.")
        return
    print(f"✅ Scraping complete. Total chunks: {len(chunks)}")

    # Step 3: Embed
    embedder = TextEmbedder()
    embeddings = embedder.embed_texts(chunks)
    if embeddings is None or len(embeddings) == 0:
        print("❌ Embedding failed. Exiting.")
        return

    # Step 4: Store in FAISS
    store = FAISSStore(dim=len(embeddings[0]))
    store.add_embeddings(embeddings, chunks)
    print("✅ Embeddings stored in FAISS index.")

    # Step 5: Accept query and search
    user_query = input("🔍 Enter your query: ")
    query_embedding = embedder.embed_texts([user_query])[0]
    results = store.search(query_embedding)

    # Step 6: Use Ollama
    context = "\n".join([r[0] for r in results])
    prompt = f"Answer based on:\n{context}\n\nQuestion: {user_query}"
    ollama_response = query_ollama(prompt)

    print("\n🤖 Ollama Response:")
    print(ollama_response)

if __name__ == "__main__":
    main()
