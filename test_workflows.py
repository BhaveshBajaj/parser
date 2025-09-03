#!/usr/bin/env python3
"""
Quick test script for Milestone 3 Agent Workflows
Run this to test the agent workflow system end-to-end.
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_result(success, message, details=None):
    icon = "✅" if success else "❌"
    print(f"{icon} {message}")
    if details:
        print(f"   {details}")

def test_server_health():
    """Test if the server is running."""
    print_section("1. TESTING SERVER HEALTH")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print_result(True, "Server is running")
            return True
        else:
            print_result(False, f"Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_result(False, "Server is not running or not accessible")
        print(f"   Error: {str(e)}")
        print(f"   Make sure to start the server with: uvicorn doc_processor.app.main:app --reload")
        return False

def upload_test_document():
    """Upload a test document and return its ID."""
    print_section("2. UPLOADING TEST DOCUMENT")
    
    # Try to find existing test files
    test_files = [
        "data/uploads/The Last Train 1_2.docx",
        "data/uploads/DB DESIGN FOR TRAINING MODULE_2.docx",
        "test_doc.txt",
        "README.md"  # Fallback to README if no test files
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            print(f"📄 Found test file: {file_path}")
            
            try:
                with open(file_path, 'rb') as f:
                    response = requests.post(
                        f"{BASE_URL}/documents",
                        files={"file": f},
                        data={"metadata": json.dumps({"test": True, "source": "test_script"})}
                    )
                
                if response.status_code == 202:
                    doc_data = response.json()
                    doc_id = doc_data["id"]
                    print_result(True, f"Document uploaded successfully", f"ID: {doc_id}")
                    return doc_id
                else:
                    print_result(False, f"Upload failed with status {response.status_code}")
                    print(f"   Response: {response.text}")
                    
            except Exception as e:
                print_result(False, f"Upload error: {str(e)}")
                
    print_result(False, "No test files found")
    print("   Create a test file or use an existing document ID")
    return None

def test_workflow(doc_id, workflow_type, context=None):
    """Test a specific workflow type."""
    if context is None:
        context = {}
    
    print(f"\n🔄 Testing {workflow_type} workflow...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/documents/{doc_id}/workflows",
            json={
                "workflow_type": workflow_type,
                "context": context
            },
            timeout=60  # Workflows can take time
        )
        
        if response.status_code == 200:
            result = response.json()
            workflow_id = result.get("workflow_id", "unknown")
            status = result.get("status", "unknown")
            
            print_result(True, f"{workflow_type} workflow completed", f"ID: {workflow_id[:8]}..., Status: {status}")
            
            # Show execution results summary
            if "execution_results" in result:
                exec_results = result["execution_results"]
                successful_agents = sum(1 for r in exec_results.values() if hasattr(r, 'get') and r.get('status') == 'completed')
                total_agents = len(exec_results)
                print(f"   📊 Agents: {successful_agents}/{total_agents} successful")
            
            # Show workflow summary
            if "workflow_summary" in result:
                summary = result["workflow_summary"]
                success_rate = summary.get("success_rate", 0)
                print(f"   📈 Success rate: {success_rate:.1%}")
            
            return True, result
        else:
            print_result(False, f"{workflow_type} workflow failed", f"Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False, None
            
    except requests.exceptions.Timeout:
        print_result(False, f"{workflow_type} workflow timed out")
        return False, None
    except Exception as e:
        print_result(False, f"{workflow_type} workflow error: {str(e)}")
        return False, None

def test_qa(doc_id, question):
    """Test Q&A functionality."""
    print(f"\n❓ Testing Q&A: '{question}'")
    
    try:
        response = requests.post(
            f"{BASE_URL}/documents/{doc_id}/qa",
            json={
                "question": question,
                "context": {"include_evidence": True}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get("answers", {})
            
            if answers:
                print_result(True, "Q&A completed")
                for agent_name, answer_data in answers.items():
                    if isinstance(answer_data, dict) and "answer" in answer_data:
                        answer = answer_data["answer"][:100] + "..." if len(answer_data["answer"]) > 100 else answer_data["answer"]
                        confidence = answer_data.get("confidence", "unknown")
                        print(f"   🤖 {agent_name}: {answer} (confidence: {confidence})")
            else:
                print_result(True, "Q&A completed but no answers found")
            
            return True
        else:
            print_result(False, f"Q&A failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_result(False, f"Q&A error: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting Agent Workflows Test Suite")
    print(f"🌐 Testing server at: {BASE_URL}")
    
    # Test 1: Server health
    if not test_server_health():
        return
    
    # Test 2: Upload document
    doc_id = upload_test_document()
    if not doc_id:
        print("\n❌ Cannot proceed without a document. Please:")
        print("   1. Create a test file (e.g., test_doc.txt)")
        print("   2. Or manually upload a document and use its ID")
        return
    
    # Test 3: Workflow execution
    print_section("3. TESTING WORKFLOWS")
    
    workflows_to_test = [
        ("summarization", {}),
        ("entity_extraction", {
            "entity_types": ["PERSON", "ORG", "GPE"],
            "confidence_threshold": 0.5
        }),
        ("full", {})
    ]
    
    successful_workflows = 0
    for workflow_type, context in workflows_to_test:
        success, result = test_workflow(doc_id, workflow_type, context)
        if success:
            successful_workflows += 1
    
    # Test 4: Q&A
    print_section("4. TESTING Q&A")
    
    questions = [
        "What is this document about?",
        "What are the main topics discussed?",
        "Who are the key people mentioned?"
    ]
    
    successful_qa = 0
    for question in questions:
        if test_qa(doc_id, question):
            successful_qa += 1
    
    # Final summary
    print_section("TEST SUMMARY")
    print(f"📄 Document ID: {doc_id}")
    print(f"🔄 Workflows: {successful_workflows}/{len(workflows_to_test)} successful")
    print(f"❓ Q&A Tests: {successful_qa}/{len(questions)} successful")
    
    if successful_workflows == len(workflows_to_test) and successful_qa == len(questions):
        print("\n🎉 All tests passed! The agent workflow system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")
    
    print(f"\n🔍 For detailed results, check the server logs or use the API docs at:")
    print(f"   {BASE_URL}/docs")

if __name__ == "__main__":
    main()
