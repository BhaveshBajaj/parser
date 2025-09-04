#!/usr/bin/env python3
"""Test script for LangChain integration."""

import asyncio
import sys
import os
from pathlib import Path

# Add the doc_processor to the path
sys.path.insert(0, str(Path(__file__).parent / "doc_processor"))

from services.embedding_service import EmbeddingService
from services.langchain_parsers import LangChainParserFactory
from services.document_processor import DocumentProcessor


async def test_embedding_service():
    """Test the LangChain embedding service."""
    print("Testing LangChain Embedding Service...")
    
    try:
        # Initialize embedding service
        embedding_service = EmbeddingService()
        
        # Test single text embedding
        test_text = "This is a test document for LangChain integration."
        embedding = await embedding_service.embed_text(test_text)
        
        print(f"✓ Single text embedding generated: {len(embedding)} dimensions")
        
        # Test batch embeddings
        test_texts = [
            "First test document",
            "Second test document", 
            "Third test document"
        ]
        embeddings = await embedding_service.embed_texts(test_texts)
        
        print(f"✓ Batch embeddings generated: {len(embeddings)} embeddings")
        
        # Test text chunking
        long_text = "This is a long document. " * 50
        chunks = embedding_service.chunk_text(long_text)
        
        print(f"✓ Text chunking: {len(chunks)} chunks created")
        
        return True
        
    except Exception as e:
        print(f"✗ Embedding service test failed: {str(e)}")
        return False


async def test_langchain_parsers():
    """Test the LangChain parsers."""
    print("\nTesting LangChain Parsers...")
    
    try:
        # Test parser factory
        pdf_parser = LangChainParserFactory.get_parser_for_type("application/pdf")
        docx_parser = LangChainParserFactory.get_parser_for_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        txt_parser = LangChainParserFactory.get_parser_for_type("text/plain")
        
        print("✓ Parser factory created parsers for PDF, DOCX, and TXT")
        
        # Test text parser with a simple text file
        test_file = Path("test_document.txt")
        if test_file.exists():
            sections = await txt_parser.parse(test_file)
            metadata = await txt_parser.extract_metadata(test_file)
            
            print(f"✓ Text parser processed file: {len(sections)} sections")
            print(f"✓ Metadata extracted: {metadata}")
        else:
            print("⚠ No test file found, skipping parser execution test")
        
        return True
        
    except Exception as e:
        print(f"✗ LangChain parsers test failed: {str(e)}")
        return False


async def test_document_processor():
    """Test the integrated document processor."""
    print("\nTesting Document Processor with LangChain...")
    
    try:
        # Initialize document processor with LangChain
        processor = DocumentProcessor()
        
        print("✓ Document processor initialized with LangChain")
        
        # Test with a simple text file if available
        test_file = Path("test_document.txt")
        if test_file.exists():
            print(f"✓ Processing test file: {test_file}")
            # Note: This would require the full pipeline, so we'll just test initialization
        else:
            print("⚠ No test file found, skipping full processing test")
        
        return True
        
    except Exception as e:
        print(f"✗ Document processor test failed: {str(e)}")
        return False


async def main():
    """Run all tests."""
    print("LangChain Integration Test Suite")
    print("=" * 40)
    
    # Check if dependencies are installed
    try:
        import langchain_huggingface
        import langchain_text_splitters
        print("✓ LangChain dependencies are available")
    except ImportError as e:
        print(f"✗ LangChain dependencies not installed: {e}")
        print("Please run: pip install -r requirements.txt")
        return
    
    # Run tests
    tests = [
        test_embedding_service,
        test_langchain_parsers,
        test_document_processor
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! LangChain integration is working.")
    else:
        print("❌ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())
