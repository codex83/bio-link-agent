"""
PubMed MCP Server

This server exposes PubMed as an MCP tool, allowing agents to:
1. Search for articles by keyword
2. Fetch article abstracts
3. Retrieve detailed article metadata
"""

import requests
import xmltodict
from typing import List, Dict, Optional
from loguru import logger
import time


class PubMedServer:
    """MCP Server for PubMed API access."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def __init__(self, email: Optional[str] = None, rate_limit: float = 0.34):
        """
        Initialize PubMed server.
        
        Args:
            email: Optional email for NCBI (increases rate limit)
            rate_limit: Seconds between requests (default 0.34s = ~3 requests/sec)
        """
        self.email = email
        self.rate_limit = rate_limit
        self.last_request_time = 0
        logger.info("PubMed MCP Server initialized")
    
    def _rate_limit_wait(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def search(self, query: str, max_results: int = 20, sort: str = "relevance") -> List[str]:
        """
        Search PubMed for articles matching the query.
        
        Args:
            query: Search query (e.g., "glioblastoma immunotherapy")
            max_results: Maximum number of results to return
            sort: Sort order ("relevance" or "pub_date")
        
        Returns:
            List of PubMed IDs (PMIDs)
        """
        self._rate_limit_wait()
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": sort,
        }
        
        if self.email:
            params["email"] = self.email
        
        try:
            response = requests.get(f"{self.BASE_URL}esearch.fcgi", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"Found {len(pmids)} articles for query: {query}")
            return pmids
        
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    def fetch_abstracts(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch full abstracts and metadata for a list of PMIDs.
        
        Args:
            pmids: List of PubMed IDs
        
        Returns:
            List of article dictionaries with title, abstract, authors, journal, etc.
        """
        if not pmids:
            return []
        
        self._rate_limit_wait()
        
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        
        if self.email:
            params["email"] = self.email
        
        try:
            response = requests.get(f"{self.BASE_URL}efetch.fcgi", params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML
            data = xmltodict.parse(response.text)
            articles = []
            
            # Handle single article vs multiple articles
            pubmed_articles = data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
            if not isinstance(pubmed_articles, list):
                pubmed_articles = [pubmed_articles]
            
            for article in pubmed_articles:
                try:
                    medline_citation = article.get("MedlineCitation", {})
                    article_data = medline_citation.get("Article", {})
                    
                    # Extract title
                    title = article_data.get("ArticleTitle", "")
                    
                    # Extract abstract
                    abstract_data = article_data.get("Abstract", {})
                    abstract_texts = abstract_data.get("AbstractText", [])
                    if not isinstance(abstract_texts, list):
                        abstract_texts = [abstract_texts]
                    
                    # Combine abstract parts
                    abstract = ""
                    for text_part in abstract_texts:
                        if isinstance(text_part, dict):
                            label = text_part.get("@Label", "")
                            text = text_part.get("#text", "")
                            abstract += f"{label}: {text}\n" if label else f"{text}\n"
                        else:
                            abstract += f"{text_part}\n"
                    
                    # Extract authors
                    author_list = article_data.get("AuthorList", {}).get("Author", [])
                    if not isinstance(author_list, list):
                        author_list = [author_list]
                    
                    authors = []
                    for author in author_list:
                        if isinstance(author, dict):
                            last_name = author.get("LastName", "")
                            fore_name = author.get("ForeName", "")
                            authors.append(f"{fore_name} {last_name}".strip())
                    
                    # Extract journal info
                    journal = article_data.get("Journal", {})
                    journal_title = journal.get("Title", "")
                    pub_date = journal.get("JournalIssue", {}).get("PubDate", {})
                    year = pub_date.get("Year", "")
                    
                    # PMID
                    pmid = medline_citation.get("PMID", {})
                    if isinstance(pmid, dict):
                        pmid = pmid.get("#text", "")
                    
                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract.strip(),
                        "authors": authors,
                        "journal": journal_title,
                        "year": year,
                    })
                
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue
            
            logger.info(f"Successfully fetched {len(articles)} abstracts")
            return articles
        
        except Exception as e:
            logger.error(f"Error fetching abstracts: {e}")
            return []
    
    def search_and_fetch(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Convenience method: Search and fetch abstracts in one call.
        
        Args:
            query: Search query
            max_results: Maximum number of results
        
        Returns:
            List of article dictionaries
        """
        pmids = self.search(query, max_results)
        return self.fetch_abstracts(pmids)


# Example usage
if __name__ == "__main__":
    # Test the server
    server = PubMedServer()
    
    # Search for glioblastoma articles
    articles = server.search_and_fetch("glioblastoma immunotherapy", max_results=5)
    
    for i, article in enumerate(articles, 1):
        print(f"\n{i}. {article['title']}")
        print(f"   PMID: {article['pmid']}")
        print(f"   Journal: {article['journal']} ({article['year']})")
        print(f"   Abstract: {article['abstract'][:200]}...")

