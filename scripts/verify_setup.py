"""
Setup Verification Script

Run this script to verify that Bio-Link Agent is properly installed and configured.
"""

import sys
import os

def check_python_version():
    """Check if Python version is 3.9+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python version: {version.major}.{version.minor}.{version.micro} (need 3.9+)")
        return False

def check_imports():
    """Check if all required packages are installed"""
    packages = [
        ("requests", "Requests"),
        ("networkx", "NetworkX"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("ollama", "Ollama"),
        ("loguru", "Loguru"),
        ("xmltodict", "xmltodict"),
    ]
    
    all_ok = True
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"✓ {display_name} installed")
        except ImportError:
            print(f"✗ {display_name} NOT installed")
            all_ok = False
    
    return all_ok

def check_ollama():
    """Check if Ollama is accessible"""
    try:
        import ollama
        # Try to list models
        models = ollama.list()
        print(f"✓ Ollama is running")
        
        # Check for required models - handle different API response formats
        model_list = models.get('models', []) if isinstance(models, dict) else models.models if hasattr(models, 'models') else []
        
        model_names = []
        for m in model_list:
            if isinstance(m, dict):
                model_names.append(m.get('name', '') or m.get('model', ''))
            elif hasattr(m, 'name'):
                model_names.append(m.name)
            elif hasattr(m, 'model'):
                model_names.append(m.model)
            else:
                model_names.append(str(m))
        
        has_model = False
        for model in model_names:
            if 'gemma' in model.lower() or 'llama' in model.lower():
                print(f"✓ Found model: {model}")
                has_model = True
                break
        
        if not has_model:
            print("⚠ No compatible model found. Run: ollama pull gemma3:12b")
            return False
        
        return True
    
    except Exception as e:
        print(f"✗ Ollama not accessible: {e}")
        print("  Make sure Ollama is installed and running")
        print("  Install: brew install ollama")
        print("  Then: ollama pull gemma3:12b")
        return False

def check_project_structure():
    """Check if all project directories exist"""
    dirs = [
        "mcp_servers",
        "agents",
        "utils",
        "demos",
        "data",
    ]
    
    all_ok = True
    for dirname in dirs:
        if os.path.isdir(dirname):
            print(f"✓ Directory exists: {dirname}/")
        else:
            print(f"✗ Directory missing: {dirname}/")
            all_ok = False
    
    return all_ok

def check_config():
    """Check if config file exists and is valid"""
    try:
        import config
        print(f"✓ Config file loaded")
        print(f"  Model: {config.OLLAMA_MODEL}")
        print(f"  Embedding: {config.EMBEDDING_MODEL}")
        return True
    except Exception as e:
        print(f"✗ Config file error: {e}")
        return False

def test_basic_functionality():
    """Test basic MCP server functionality"""
    try:
        from mcp_servers.pubmed_server import PubMedServer
        from mcp_servers.clinicaltrials_server import ClinicalTrialsServer
        
        print("\n" + "="*60)
        print("Testing API Connectivity...")
        print("="*60)
        
        # Test PubMed
        print("\nTesting PubMed API...")
        pubmed = PubMedServer()
        pmids = pubmed.search("diabetes", max_results=2)
        
        if pmids:
            print(f"✓ PubMed API working (found {len(pmids)} articles)")
        else:
            print("⚠ PubMed API returned no results")
        
        # Test ClinicalTrials.gov
        print("\nTesting ClinicalTrials.gov API...")
        trials = ClinicalTrialsServer()
        nct_ids = trials.search("diabetes", max_results=2)
        
        if nct_ids:
            print(f"✓ ClinicalTrials.gov API working (found {len(nct_ids)} trials)")
        else:
            print("⚠ ClinicalTrials.gov API returned no results")
        
        return True
    
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False

def main():
    """Run all verification checks"""
    print("\n" + "="*60)
    print("BIO-LINK AGENT - SETUP VERIFICATION")
    print("="*60)
    
    print("\n1. Checking Python Version...")
    print("-" * 60)
    py_ok = check_python_version()
    
    print("\n2. Checking Python Packages...")
    print("-" * 60)
    packages_ok = check_imports()
    
    print("\n3. Checking Ollama...")
    print("-" * 60)
    ollama_ok = check_ollama()
    
    print("\n4. Checking Project Structure...")
    print("-" * 60)
    structure_ok = check_project_structure()
    
    print("\n5. Checking Configuration...")
    print("-" * 60)
    config_ok = check_config()
    
    print("\n6. Testing API Connectivity...")
    print("-" * 60)
    api_ok = test_basic_functionality()
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    checks = [
        ("Python Version", py_ok),
        ("Python Packages", packages_ok),
        ("Ollama", ollama_ok),
        ("Project Structure", structure_ok),
        ("Configuration", config_ok),
        ("API Connectivity", api_ok),
    ]
    
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    
    for check_name, ok in checks:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{status:>10}  {check_name}")
    
    print("-" * 60)
    print(f"Result: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! You're ready to run the demos.")
        print("\nNext steps:")
        print("  1. Run semantic matcher demo: python demos/demo_matcher.py")
        print("  2. Run landscape agent demo: python demos/demo_landscape.py")
        print("  3. Read QUICKSTART.md for more information")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Install Ollama: brew install ollama")
        print("  - Pull model: ollama pull gemma3:12b")
    
    print("\n")
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

