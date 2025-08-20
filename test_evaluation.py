"""
Tests for evaluation functionality
"""
import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import ChatbotEvaluator
from config import config
from database import init_database, get_session, EvaluationResult

class TestChatbotEvaluator:
    """Test cases for ChatbotEvaluator"""
    
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Setup test environment"""
        # Use test database
        config.database_url = "sqlite:///./test_evaluation.db"
        init_database()
        
        # Mock API key if not set
        if not config.openai_api_key:
            config.openai_api_key = "test-key"
        
        yield
        
        # Cleanup
        test_db_path = Path("test_evaluation.db")
        if test_db_path.exists():
            test_db_path.unlink()
    
    @patch('evaluation.ChatOpenAI')
    @patch('evaluation.load_evaluator')
    @patch('evaluation.get_chatbot')
    def test_evaluator_initialization(self, mock_get_chatbot, mock_load_evaluator, mock_llm):
        """Test evaluator initialization"""
        mock_get_chatbot.return_value = Mock()
        mock_load_evaluator.return_value = Mock()
        mock_llm.return_value = Mock()
        
        evaluator = ChatbotEvaluator()
        
        assert evaluator.chatbot is not None
        assert evaluator.llm is not None
        assert evaluator.faithfulness_evaluator is not None
        assert evaluator.relevance_evaluator is not None
    
    def test_sample_dataset_creation(self):
        """Test sample dataset creation"""
        with patch('evaluation.ChatOpenAI'), patch('evaluation.load_evaluator'), patch('evaluation.get_chatbot'):
            evaluator = ChatbotEvaluator()
            dataset = evaluator.create_sample_dataset()
            
            assert len(dataset) > 0
            
            # Check required fields
            for item in dataset:
                assert "question" in item
                assert "expected_answer" in item
                assert "expected_tool" in item
                assert "category" in item
            
            # Check that different tool types are represented
            tools_in_dataset = {item["expected_tool"] for item in dataset}
            assert "rag" in tools_in_dataset
            assert "math" in tools_in_dataset
            assert "web_search" in tools_in_dataset
    
    def test_load_test_dataset_nonexistent(self):
        """Test loading test dataset when file doesn't exist"""
        with patch('evaluation.ChatOpenAI'), patch('evaluation.load_evaluator'), patch('evaluation.get_chatbot'):
            evaluator = ChatbotEvaluator()
            
            # Try to load from non-existent path
            nonexistent_path = Path("nonexistent_dataset.json")
            dataset = evaluator.load_test_dataset(nonexistent_path)
            
            # Should return sample dataset
            assert len(dataset) > 0
            assert isinstance(dataset, list)
    
    def test_load_test_dataset_existing(self):
        """Test loading test dataset from existing file"""
        with patch('evaluation.ChatOpenAI'), patch('evaluation.load_evaluator'), patch('evaluation.get_chatbot'):
            evaluator = ChatbotEvaluator()
            
            # Create a test dataset file
            test_dataset = [
                {
                    "question": "Test question",
                    "expected_answer": "Test answer",
                    "expected_tool": "rag",
                    "category": "test"
                }
            ]
            
            test_file_path = Path("test_dataset.json")
            with open(test_file_path, 'w') as f:
                json.dump(test_dataset, f)
            
            try:
                dataset = evaluator.load_test_dataset(test_file_path)
                
                assert len(dataset) == 1
                assert dataset[0]["question"] == "Test question"
                assert dataset[0]["expected_tool"] == "rag"
            finally:
                # Cleanup
                if test_file_path.exists():
                    test_file_path.unlink()
    
    def test_faithfulness_evaluation_no_sources(self):
        """Test faithfulness evaluation with no sources"""
        with patch('evaluation.ChatOpenAI'), patch('evaluation.load_evaluator'), patch('evaluation.get_chatbot'):
            evaluator = ChatbotEvaluator()
            
            result = evaluator.evaluate_faithfulness(
                "Test question",
                "Test answer", 
                []
            )
            
            assert result["score"] == "N/A"
            assert "No sources provided" in result["reasoning"]
    
    def test_faithfulness_evaluation_with_sources(self):
        """Test faithfulness evaluation with sources"""
        with patch('evaluation.ChatOpenAI'), patch('evaluation.load_evaluator'), patch('evaluation.get_chatbot'):
            # Mock the evaluator
            mock_evaluator = Mock()
            mock_evaluator.evaluate_strings.return_value = {
                "score": 4,
                "reasoning": "The answer is faithful to the sources"
            }
            
            evaluator = ChatbotEvaluator()
            evaluator.faithfulness_evaluator = mock_evaluator
            
            sources = [
                {"file": "test.pdf", "page": 1},
                {"source": "web_search"}
            ]
            
            result = evaluator.evaluate_faithfulness(
                "Test question",
                "Test answer",
                sources
            )
            
            assert result["score"] == 4
            assert "faithful" in result["reasoning"]
    
    def test_relevance_evaluation(self):
        """Test relevance evaluation"""
        with patch('evaluation.ChatOpenAI'), patch('evaluation.load_evaluator'), patch('evaluation.get_chatbot'):
            # Mock the evaluator
            mock_evaluator = Mock()
            mock_evaluator.evaluate_strings.return_value = {
                "score": 5,
                "reasoning": "The answer is highly relevant"
            }
            
            evaluator = ChatbotEvaluator()
            evaluator.relevance_evaluator = mock_evaluator
            
            result = evaluator.evaluate_relevance(
                "What is AI?",
                "AI stands for Artificial Intelligence"
            )
            
            assert result["score"] == 5
            assert "relevant" in result["reasoning"]
    
    def test_tool_routing_evaluation(self):
        """Test tool routing evaluation"""
        with patch('evaluation.ChatOpenAI'), patch('evaluation.load_evaluator'), patch('evaluation.get_chatbot'):
            evaluator = ChatbotEvaluator()
            
            # Test correct routing
            result = evaluator.evaluate_tool_routing(
                "Calculate 2+2",
                "math",
                "math"
            )
            
            assert result["correct"] is True
            assert result["expected"] == "math"
            assert result["actual"] == "math"
            assert result["score"] == 1.0
            
            # Test incorrect routing
            result = evaluator.evaluate_tool_routing(
                "Calculate 2+2",
                "math",
                "rag"
            )
            
            assert result["correct"] is False
            assert result["expected"] == "math"
            assert result["actual"] == "rag"
            assert result["score"] == 0.0
    
    def test_summary_stats_calculation(self):
        """Test summary statistics calculation"""
        with patch('evaluation.ChatOpenAI'), patch('evaluation.load_evaluator'), patch('evaluation.get_chatbot'):
            evaluator = ChatbotEvaluator()
            
            # Mock data
            scores = {
                "faithfulness": [4, 5, 3, 4],
                "relevance": [5, 4, 5, 4],
                "tool_routing": [1.0, 1.0, 0.0, 1.0]
            }
            
            results = [
                {
                    "category": "math",
                    "tool_routing": {"correct": True}
                },
                {
                    "category": "math",
                    "tool_routing": {"correct": True}
                },
                {
                    "category": "rag",
                    "tool_routing": {"correct": False}
                },
                {
                    "category": "web_search",
                    "tool_routing": {"correct": True}
                }
            ]
            
            summary = evaluator._calculate_summary_stats(scores, results)
            
            # Check basic stats
            assert summary["faithfulness"]["mean"] == 4.0
            assert summary["relevance"]["mean"] == 4.5
            assert summary["tool_routing"]["mean"] == 0.75
            
            # Check category performance
            assert "math" in summary["category_performance"]
            assert summary["category_performance"]["math"]["accuracy"] == 1.0
            assert summary["category_performance"]["rag"]["accuracy"] == 0.0
    
    @patch('evaluation.ChatOpenAI')
    @patch('evaluation.load_evaluator')
    @patch('evaluation.get_chatbot')
    def test_evaluation_result_saving(self, mock_get_chatbot, mock_load_evaluator, mock_llm):
        """Test saving evaluation results to database"""
        mock_get_chatbot.return_value = Mock()
        mock_load_evaluator.return_value = Mock()
        mock_llm.return_value = Mock()
        
        evaluator = ChatbotEvaluator()
        
        # Mock evaluation result
        result = {
            "question": "Test question",
            "expected_answer": "Test expected",
            "actual_answer": "Test actual",
            "tool_used": "rag",
            "sources": [{"file": "test.pdf"}],
            "faithfulness": {"score": 4},
            "relevance": {"score": 5}
        }
        
        evaluator._save_evaluation_result(result)
        
        # Check if saved to database
        db = next(get_session())
        eval_results = db.query(EvaluationResult).all()
        assert len(eval_results) == 1
        
        saved_result = eval_results[0]
        assert saved_result.question == "Test question"
        assert saved_result.tool_used == "rag"
        assert saved_result.faithfulness_score == "4"
        assert saved_result.relevance_score == "5"
        db.close()
    
    def test_evaluation_dashboard_data(self):
        """Test creating evaluation dashboard data"""
        with patch('evaluation.ChatOpenAI'), patch('evaluation.load_evaluator'), patch('evaluation.get_chatbot'):
            evaluator = ChatbotEvaluator()
            
            # Add some test data to database
            db = next(get_session())
            test_result = EvaluationResult(
                question="Test question for dashboard",
                expected_answer="Test expected",
                actual_answer="Test actual",
                tool_used="rag",
                sources=[{"file": "test.pdf"}],
                faithfulness_score="4",
                relevance_score="5"
            )
            db.add(test_result)
            db.commit()
            db.close()
            
            # Get dashboard data
            df = evaluator.create_evaluation_dashboard()
            
            assert len(df) == 1
            assert "Question" in df.columns
            assert "Tool Used" in df.columns
            assert "Faithfulness" in df.columns
            assert "Relevance" in df.columns

class TestEvaluationIntegration:
    """Integration tests for evaluation system"""
    
    @pytest.fixture(autouse=True)
    def setup_integration_env(self):
        """Setup integration test environment"""
        config.database_url = "sqlite:///./test_eval_integration.db"
        init_database()
        
        yield
        
        # Cleanup
        test_db_path = Path("test_eval_integration.db")
        if test_db_path.exists():
            test_db_path.unlink()
    
    @patch('evaluation.ChatOpenAI')
    @patch('evaluation.load_evaluator')
    @patch('evaluation.get_chatbot')
    def test_end_to_end_evaluation(self, mock_get_chatbot, mock_load_evaluator, mock_llm):
        """Test end-to-end evaluation flow"""
        # Mock chatbot
        mock_chatbot = Mock()
        mock_chatbot.create_session.return_value = "test-session"
        mock_chatbot.chat.return_value = {
            "response": "Test response",
            "sources": [{"file": "test.pdf", "page": 1}],
            "tool_used": "rag"
        }
        mock_get_chatbot.return_value = mock_chatbot
        
        # Mock evaluators
        mock_faithfulness = Mock()
        mock_faithfulness.evaluate_strings.return_value = {"score": 4, "reasoning": "Good"}
        
        mock_relevance = Mock()
        mock_relevance.evaluate_strings.return_value = {"score": 5, "reasoning": "Excellent"}
        
        mock_load_evaluator.side_effect = [mock_faithfulness, mock_relevance]
        mock_llm.return_value = Mock()
        
        evaluator = ChatbotEvaluator()
        
        # Create small test dataset
        test_dataset = [
            {
                "question": "What is AI?",
                "expected_answer": "AI is artificial intelligence",
                "expected_tool": "rag",
                "category": "definition"
            }
        ]
        
        # Run evaluation
        report = evaluator.run_evaluation(test_dataset)
        
        # Check report structure
        assert "timestamp" in report
        assert "total_test_cases" in report
        assert "successful_evaluations" in report
        assert "summary" in report
        assert "detailed_results" in report
        
        assert report["total_test_cases"] == 1
        assert report["successful_evaluations"] == 1
        
        # Check detailed results
        result = report["detailed_results"][0]
        assert result["question"] == "What is AI?"
        assert result["actual_answer"] == "Test response"
        assert result["tool_used"] == "rag"
        assert result["faithfulness"]["score"] == 4
        assert result["relevance"]["score"] == 5
        assert result["tool_routing"]["correct"] is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])