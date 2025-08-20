"""
Evaluation module for testing chatbot performance
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import pandas as pd

# LangChain evaluation imports
from langchain.evaluation import load_evaluator
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

# Local imports
from config import config
from chatbot import get_chatbot
from database import get_session, EvaluationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotEvaluator:
    """Evaluator for the chatbot performance"""
    
    def __init__(self):
        self.chatbot = get_chatbot()
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=0,
            openai_api_key=config.openai_api_key
        )
        
        # Load evaluators
        self.faithfulness_evaluator = load_evaluator(
            "labeled_score_string",
            criteria={
                "faithfulness": "Does the submission faithfully represent the information from the reference?"
            },
            llm=self.llm
        )
        
        self.relevance_evaluator = load_evaluator(
            "labeled_score_string", 
            criteria={
                "relevance": "Is the submission relevant to the input question?"
            },
            llm=self.llm
        )
    
    def load_test_dataset(self, dataset_path: Path = None) -> List[Dict[str, Any]]:
        """Load test dataset from JSON file"""
        if dataset_path is None:
            dataset_path = config.eval_dataset_path
        
        if not dataset_path.exists():
            logger.warning(f"Test dataset not found at {dataset_path}")
            return self.create_sample_dataset()
        
        try:
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            logger.info(f"Loaded {len(dataset)} test cases from {dataset_path}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return self.create_sample_dataset()
    
    def create_sample_dataset(self) -> List[Dict[str, Any]]:
        """Create a sample test dataset"""
        sample_dataset = [
            {
                "question": "What is GDPR?",
                "expected_answer": "GDPR is the General Data Protection Regulation, a comprehensive data protection law in the EU.",
                "expected_tool": "rag",
                "category": "policy"
            },
            {
                "question": "Calculate 15% of 250000",
                "expected_answer": "37500",
                "expected_tool": "math",
                "category": "calculation"
            },
            {
                "question": "What are the latest AI safety guidelines?",
                "expected_answer": "Recent AI safety guidelines focus on alignment, transparency, and robustness.",
                "expected_tool": "web_search",
                "category": "current_events"
            },
            {
                "question": "What does the AI Act say about high-risk AI systems?",
                "expected_answer": "The AI Act classifies certain AI systems as high-risk and requires specific compliance measures.",
                "expected_tool": "rag", 
                "category": "policy"
            },
            {
                "question": "What is 2^10?",
                "expected_answer": "1024",
                "expected_tool": "math",
                "category": "calculation"
            },
            {
                "question": "Recent developments in AI regulation 2024",
                "expected_answer": "2024 has seen significant developments in AI regulation globally.",
                "expected_tool": "web_search",
                "category": "current_events"
            }
        ]
        
        # Save sample dataset
        with open(config.eval_dataset_path, 'w') as f:
            json.dump(sample_dataset, f, indent=2)
        
        logger.info(f"Created sample dataset with {len(sample_dataset)} test cases")
        return sample_dataset
    
    def evaluate_faithfulness(self, question: str, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """Evaluate faithfulness of answer to sources"""
        if not sources:
            return {"score": "N/A", "reasoning": "No sources provided"}
        
        try:
            # Create reference text from sources
            reference_texts = []
            for source in sources:
                if "file" in source:
                    reference_texts.append(f"Source: {source['file']}")
                elif "source" in source:
                    reference_texts.append(f"Source: {source['source']}")
            
            reference = "\n".join(reference_texts)
            
            if not reference.strip():
                return {"score": "N/A", "reasoning": "No valid source content"}
            
            result = self.faithfulness_evaluator.evaluate_strings(
                prediction=answer,
                reference=reference,
                input=question
            )
            
            return {
                "score": result.get("score", "N/A"),
                "reasoning": result.get("reasoning", "No reasoning provided")
            }
            
        except Exception as e:
            logger.error(f"Faithfulness evaluation error: {e}")
            return {"score": "ERROR", "reasoning": str(e)}
    
    def evaluate_relevance(self, question: str, answer: str) -> Dict[str, Any]:
        """Evaluate relevance of answer to question"""
        try:
            result = self.relevance_evaluator.evaluate_strings(
                prediction=answer,
                input=question
            )
            
            return {
                "score": result.get("score", "N/A"),
                "reasoning": result.get("reasoning", "No reasoning provided")
            }
            
        except Exception as e:
            logger.error(f"Relevance evaluation error: {e}")
            return {"score": "ERROR", "reasoning": str(e)}
    
    def evaluate_tool_routing(self, question: str, expected_tool: str, actual_tool: str) -> Dict[str, Any]:
        """Evaluate if the correct tool was used"""
        correct = expected_tool.lower() == actual_tool.lower()
        
        return {
            "correct": correct,
            "expected": expected_tool,
            "actual": actual_tool,
            "score": 1.0 if correct else 0.0
        }
    
    def run_evaluation(self, dataset: List[Dict[str, Any]] = None, session_id: str = None) -> Dict[str, Any]:
        """Run complete evaluation on dataset"""
        if dataset is None:
            dataset = self.load_test_dataset()
        
        if session_id is None:
            session_id = self.chatbot.create_session("evaluation_user")
        
        logger.info(f"Starting evaluation with {len(dataset)} test cases")
        
        results = []
        scores = {
            "faithfulness": [],
            "relevance": [], 
            "tool_routing": []
        }
        
        for i, test_case in enumerate(dataset, 1):
            logger.info(f"Evaluating test case {i}/{len(dataset)}: {test_case['question'][:50]}...")
            
            try:
                # Get chatbot response
                response = self.chatbot.chat(test_case["question"], session_id)
                
                # Evaluate faithfulness
                faithfulness_result = self.evaluate_faithfulness(
                    test_case["question"],
                    response["response"], 
                    response["sources"]
                )
                
                # Evaluate relevance
                relevance_result = self.evaluate_relevance(
                    test_case["question"],
                    response["response"]
                )
                
                # Evaluate tool routing
                routing_result = self.evaluate_tool_routing(
                    test_case["question"],
                    test_case.get("expected_tool", "rag"),
                    response["tool_used"]
                )
                
                # Store result
                result = {
                    "question": test_case["question"],
                    "expected_answer": test_case.get("expected_answer", ""),
                    "actual_answer": response["response"],
                    "tool_used": response["tool_used"],
                    "sources": response["sources"],
                    "faithfulness": faithfulness_result,
                    "relevance": relevance_result,
                    "tool_routing": routing_result,
                    "category": test_case.get("category", "unknown")
                }
                
                results.append(result)
                
                # Collect scores
                if faithfulness_result["score"] not in ["N/A", "ERROR"]:
                    scores["faithfulness"].append(float(faithfulness_result["score"]))
                
                if relevance_result["score"] not in ["N/A", "ERROR"]:
                    scores["relevance"].append(float(relevance_result["score"]))
                
                scores["tool_routing"].append(routing_result["score"])
                
                # Save to database
                self._save_evaluation_result(result)
                
            except Exception as e:
                logger.error(f"Error evaluating test case {i}: {e}")
                error_result = {
                    "question": test_case["question"],
                    "error": str(e)
                }
                results.append(error_result)
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(scores, results)
        
        # Save evaluation report
        evaluation_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_test_cases": len(dataset),
            "successful_evaluations": len([r for r in results if "error" not in r]),
            "summary": summary,
            "detailed_results": results
        }
        
        self._save_evaluation_report(evaluation_report)
        
        logger.info("✅ Evaluation completed successfully")
        return evaluation_report
    
    def _calculate_summary_stats(self, scores: Dict[str, List], results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        summary = {}
        
        for metric, score_list in scores.items():
            if score_list:
                summary[metric] = {
                    "mean": sum(score_list) / len(score_list),
                    "count": len(score_list),
                    "scores": score_list
                }
            else:
                summary[metric] = {
                    "mean": 0.0,
                    "count": 0,
                    "scores": []
                }
        
        # Tool routing accuracy by category
        categories = {}
        for result in results:
            if "error" not in result:
                category = result.get("category", "unknown")
                if category not in categories:
                    categories[category] = {"correct": 0, "total": 0}
                
                categories[category]["total"] += 1
                if result["tool_routing"]["correct"]:
                    categories[category]["correct"] += 1
        
        # Calculate accuracy per category
        category_accuracy = {}
        for category, stats in categories.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            category_accuracy[category] = {
                "accuracy": accuracy,
                "correct": stats["correct"],
                "total": stats["total"]
            }
        
        summary["category_performance"] = category_accuracy
        
        return summary
    
    def _save_evaluation_result(self, result: Dict[str, Any]):
        """Save individual evaluation result to database"""
        if "error" in result:
            return
        
        db = next(get_session())
        try:
            eval_result = EvaluationResult(
                question=result["question"],
                expected_answer=result["expected_answer"],
                actual_answer=result["actual_answer"],
                tool_used=result["tool_used"],
                sources=result["sources"],
                faithfulness_score=str(result["faithfulness"]["score"]),
                groundedness_score="N/A",  # Could add groundedness evaluation
                relevance_score=str(result["relevance"]["score"])
            )
            db.add(eval_result)
            db.commit()
        except Exception as e:
            logger.error(f"Error saving evaluation result: {e}")
            db.rollback()
        finally:
            db.close()
    
    def _save_evaluation_report(self, report: Dict[str, Any]):
        """Save evaluation report to file"""
        report_path = config.data_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Evaluation report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}")
    
    def create_evaluation_dashboard(self) -> pd.DataFrame:
        """Create evaluation dashboard data"""
        db = next(get_session())
        try:
            results = db.query(EvaluationResult).all()
            
            if not results:
                return pd.DataFrame()
            
            data = []
            for result in results:
                data.append({
                    "Question": result.question[:100] + "..." if len(result.question) > 100 else result.question,
                    "Tool Used": result.tool_used,
                    "Faithfulness": result.faithfulness_score,
                    "Relevance": result.relevance_score,
                    "Timestamp": result.evaluation_timestamp
                })
            
            return pd.DataFrame(data)
            
        finally:
            db.close()

def main():
    """CLI for running evaluations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run chatbot evaluation")
    parser.add_argument("--dataset", type=Path, help="Path to test dataset JSON file")
    parser.add_argument("--create-sample", action="store_true", help="Create sample dataset")
    parser.add_argument("--output", type=Path, help="Output directory for results")
    
    args = parser.parse_args()
    
    evaluator = ChatbotEvaluator()
    
    if args.create_sample:
        dataset = evaluator.create_sample_dataset()
        print(f"✅ Created sample dataset with {len(dataset)} test cases")
        return
    
    try:
        # Load dataset
        dataset = evaluator.load_test_dataset(args.dataset)
        
        # Run evaluation
        report = evaluator.run_evaluation(dataset)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        summary = report["summary"]
        print(f"Total test cases: {report['total_test_cases']}")
        print(f"Successful evaluations: {report['successful_evaluations']}")
        
        for metric, stats in summary.items():
            if metric != "category_performance" and isinstance(stats, dict):
                print(f"{metric.title()} - Mean: {stats['mean']:.2f} (n={stats['count']})")
        
        print("\nCategory Performance:")
        for category, stats in summary.get("category_performance", {}).items():
            print(f"  {category}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()