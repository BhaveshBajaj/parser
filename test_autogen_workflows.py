#!/usr/bin/env python3
"""
Test script for LLM-only AutoGen workflow implementation.
This script tests the AutoGen-based document processing workflows.
"""

import asyncio
import json
import requests
import time
from pathlib import Path
from typing import Dict, Any, List

BASE_URL = "http://localhost:8000"

class WorkflowTester:
    """Test class for LLM-powered AutoGen workflows."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.test_results = []
        
    def test_health_check(self) -> bool:
        """Test if the API is running."""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    def upload_test_document(self, file_path: str) -> str:
        """Upload a test document and return its ID."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Test file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/documents",
                files={"file": f},
                data={"metadata": json.dumps({"test": True, "autogen_test": True})}
            )
        
        if response.status_code == 202:
            return response.json()["id"]
        else:
            raise Exception(f"Failed to upload document: {response.text}")
    
    def test_legacy_workflow(self, document_id: str, workflow_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test legacy workflow orchestrator."""
        start_time = time.time()
        
        response = requests.post(
            f"{self.base_url}/documents/{document_id}/workflows",
            json={
                "workflow_type": workflow_type,
                "context": context or {}
            }
        )
        
        end_time = time.time()
        
        result = {
            "orchestrator": "legacy",
            "workflow_type": workflow_type,
            "status_code": response.status_code,
            "execution_time": end_time - start_time,
            "response": response.json() if response.status_code == 200 else response.text
        }
        
        return result
    
    def test_autogen_workflow(self, document_id: str, workflow_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test AutoGen workflow orchestrator."""
        start_time = time.time()
        
        response = requests.post(
            f"{self.base_url}/documents/{document_id}/autogen-workflows",
            json={
                "workflow_type": workflow_type,
                "context": context or {}
            }
        )
        
        end_time = time.time()
        
        result = {
            "orchestrator": "autogen",
            "workflow_type": workflow_type,
            "status_code": response.status_code,
            "execution_time": end_time - start_time,
            "response": response.json() if response.status_code == 200 else response.text
        }
        
        return result
    
    def test_legacy_qa(self, document_id: str, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test legacy Q&A."""
        start_time = time.time()
        
        response = requests.post(
            f"{self.base_url}/documents/{document_id}/qa",
            json={
                "question": question,
                "context": context or {}
            }
        )
        
        end_time = time.time()
        
        return {
            "orchestrator": "legacy",
            "type": "qa",
            "question": question,
            "status_code": response.status_code,
            "execution_time": end_time - start_time,
            "response": response.json() if response.status_code == 200 else response.text
        }
    
    def test_autogen_qa(self, document_id: str, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test AutoGen Q&A."""
        start_time = time.time()
        
        response = requests.post(
            f"{self.base_url}/documents/{document_id}/autogen-qa",
            json={
                "question": question,
                "context": context or {}
            }
        )
        
        end_time = time.time()
        
        return {
            "orchestrator": "autogen",
            "type": "qa",
            "question": question,
            "status_code": response.status_code,
            "execution_time": end_time - start_time,
            "response": response.json() if response.status_code == 200 else response.text
        }
    
    def test_orchestrator_comparison(self) -> Dict[str, Any]:
        """Test the orchestrator comparison endpoint."""
        response = requests.get(f"{self.base_url}/orchestrators/compare")
        
        return {
            "endpoint": "orchestrators/compare",
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive comparison tests."""
        print("🚀 Starting AutoGen Workflow Tests")
        print("=" * 50)
        
        # Check API health
        print("1. Checking API health...")
        if not self.test_health_check():
            print("❌ API is not running. Please start the server first.")
            return {"error": "API not available"}
        print("✅ API is running")
        
        # Upload test documents
        print("\n2. Uploading test documents...")
        test_files = [
            "data/uploads/test_doc.txt",
            "data/uploads/test_doc_2.txt"
        ]
        
        document_ids = []
        for file_path in test_files:
            try:
                if Path(file_path).exists():
                    doc_id = self.upload_test_document(file_path)
                    document_ids.append(doc_id)
                    print(f"✅ Uploaded: {file_path} -> {doc_id}")
                else:
                    print(f"⚠️  File not found: {file_path}")
            except Exception as e:
                print(f"❌ Failed to upload {file_path}: {str(e)}")
        
        if not document_ids:
            print("❌ No documents uploaded. Cannot proceed with tests.")
            return {"error": "No test documents available"}
        
        # Test workflow types
        workflow_types = ["summarization", "entity_extraction"]
        test_results = {}
        
        print(f"\n3. Testing workflows on {len(document_ids)} documents...")
        
        for workflow_type in workflow_types:
            print(f"\n   Testing {workflow_type} workflow:")
            
            # Test legacy workflow
            try:
                legacy_result = self.test_legacy_workflow(document_ids[0], workflow_type)
                print(f"   ✅ Legacy {workflow_type}: {legacy_result['execution_time']:.2f}s")
                test_results[f"legacy_{workflow_type}"] = legacy_result
            except Exception as e:
                print(f"   ❌ Legacy {workflow_type} failed: {str(e)}")
                test_results[f"legacy_{workflow_type}"] = {"error": str(e)}
            
            # Test AutoGen workflow
            try:
                autogen_result = self.test_autogen_workflow(document_ids[0], workflow_type)
                print(f"   ✅ AutoGen {workflow_type}: {autogen_result['execution_time']:.2f}s")
                test_results[f"autogen_{workflow_type}"] = autogen_result
            except Exception as e:
                print(f"   ❌ AutoGen {workflow_type} failed: {str(e)}")
                test_results[f"autogen_{workflow_type}"] = {"error": str(e)}
        
        # Test Q&A capabilities
        print(f"\n4. Testing Q&A capabilities...")
        test_questions = [
            "What is this document about?",
            "What are the main topics discussed?",
            "Who are the key people mentioned?"
        ]
        
        for question in test_questions:
            print(f"\n   Question: {question[:50]}...")
            
            # Test legacy Q&A
            try:
                legacy_qa = self.test_legacy_qa(document_ids[0], question)
                print(f"   ✅ Legacy Q&A: {legacy_qa['execution_time']:.2f}s")
                test_results[f"legacy_qa_{len(test_results)}"] = legacy_qa
            except Exception as e:
                print(f"   ❌ Legacy Q&A failed: {str(e)}")
                test_results[f"legacy_qa_{len(test_results)}"] = {"error": str(e)}
            
            # Test AutoGen Q&A
            try:
                autogen_qa = self.test_autogen_qa(document_ids[0], question)
                print(f"   ✅ AutoGen Q&A: {autogen_qa['execution_time']:.2f}s")
                test_results[f"autogen_qa_{len(test_results)}"] = autogen_qa
            except Exception as e:
                print(f"   ❌ AutoGen Q&A failed: {str(e)}")
                test_results[f"autogen_qa_{len(test_results)}"] = {"error": str(e)}
        
        # Test orchestrator comparison
        print(f"\n5. Testing orchestrator comparison...")
        try:
            comparison_result = self.test_orchestrator_comparison()
            print(f"   ✅ Comparison endpoint: Status {comparison_result['status_code']}")
            test_results["orchestrator_comparison"] = comparison_result
        except Exception as e:
            print(f"   ❌ Comparison endpoint failed: {str(e)}")
            test_results["orchestrator_comparison"] = {"error": str(e)}
        
        # Generate summary
        print(f"\n6. Test Summary:")
        print("=" * 50)
        
        successful_tests = sum(1 for result in test_results.values() 
                             if isinstance(result, dict) and "error" not in result and result.get("status_code") == 200)
        total_tests = len(test_results)
        
        print(f"✅ Successful tests: {successful_tests}/{total_tests}")
        print(f"📊 Success rate: {successful_tests/total_tests*100:.1f}%")
        
        # Performance comparison
        legacy_times = []
        autogen_times = []
        
        for key, result in test_results.items():
            if isinstance(result, dict) and "execution_time" in result:
                if "legacy" in key:
                    legacy_times.append(result["execution_time"])
                elif "autogen" in key:
                    autogen_times.append(result["execution_time"])
        
        if legacy_times and autogen_times:
            avg_legacy = sum(legacy_times) / len(legacy_times)
            avg_autogen = sum(autogen_times) / len(autogen_times)
            
            print(f"⚡ Average Legacy time: {avg_legacy:.2f}s")
            print(f"🤖 Average AutoGen time: {avg_autogen:.2f}s")
            print(f"📈 AutoGen is {avg_autogen/avg_legacy:.1f}x slower (expected due to LLM calls)")
        
        return {
            "summary": {
                "successful_tests": successful_tests,
                "total_tests": total_tests,
                "success_rate": successful_tests/total_tests,
                "average_legacy_time": sum(legacy_times) / len(legacy_times) if legacy_times else 0,
                "average_autogen_time": sum(autogen_times) / len(autogen_times) if autogen_times else 0
            },
            "detailed_results": test_results,
            "document_ids": document_ids
        }

def main():
    """Main test execution."""
    tester = WorkflowTester()
    
    print("AutoGen Workflow Implementation Test")
    print("This script tests both legacy and AutoGen workflows")
    print()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    # Save results to file
    results_file = "autogen_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Print final recommendations
    print(f"\n💡 Recommendations:")
    print("- Use Legacy orchestrator for high-throughput batch processing")
    print("- Use AutoGen orchestrator for complex analysis and interactive workflows")
    print("- Both orchestrators can run in parallel for gradual migration")
    print("- Monitor performance and quality metrics during migration")
    
    return results

if __name__ == "__main__":
    results = main()
