"""
Monitoring and metrics system for the Context-Aware Research Chatbot
"""
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import logging
from pathlib import Path

from database import get_session, ConversationLog, ChatSession, DatabaseManager
from config import config

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    active_sessions: int
    total_requests: int
    avg_response_time: float
    error_rate: float

@dataclass
class ChatbotMetrics:
    """Chatbot-specific metrics"""
    timestamp: str
    total_conversations: int
    conversations_last_hour: int
    conversations_last_day: int
    avg_conversation_length: float
    tool_usage: Dict[str, int]
    avg_response_time_by_tool: Dict[str, float]
    success_rate: float
    user_satisfaction: float

class MetricsCollector:
    """Collects and aggregates system and chatbot metrics"""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.metrics_history: deque = deque(maxlen=1440)  # 24 hours of minutes
        self.request_times: deque = deque(maxlen=1000)
        self.error_count = 0
        self.total_requests = 0
        self.active_sessions = set()
        self.is_running = False
        self.collection_thread = None
        
        # Response time tracking by tool
        self.tool_response_times = defaultdict(list)
        
        # Create metrics directory
        self.metrics_dir = Path("metrics")
        self.metrics_dir.mkdir(exist_ok=True)
    
    def start_collection(self):
        """Start metrics collection in background thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.collection_thread = threading.Thread(target=self._collect_metrics_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def _collect_metrics_loop(self):
        """Main metrics collection loop"""
        while self.is_running:
            try:
                self._collect_system_metrics()
                self._collect_chatbot_metrics()
                self._save_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Calculate error rate
        error_rate = (self.error_count / max(self.total_requests, 1)) * 100
        
        # Calculate average response time
        avg_response_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
        
        metrics = SystemMetrics(
            timestamp=datetime.utcnow().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            disk_usage_percent=disk.percent,
            active_sessions=len(self.active_sessions),
            total_requests=self.total_requests,
            avg_response_time=avg_response_time,
            error_rate=error_rate
        )
        
        self.metrics_history.append(('system', asdict(metrics)))
    
    def _collect_chatbot_metrics(self):
        """Collect chatbot-specific metrics"""
        try:
            db_manager = DatabaseManager()
            global_stats = db_manager.get_global_stats()
            
            # Get conversation stats
            db = next(get_session())
            
            # Recent conversations
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)
            
            conversations_last_hour = db.query(ConversationLog).filter(
                ConversationLog.timestamp > hour_ago
            ).count()
            
            conversations_last_day = db.query(ConversationLog).filter(
                ConversationLog.timestamp > day_ago
            ).count()
            
            # Average conversation length
            session_lengths = db.query(ChatSession.session_id).join(ConversationLog).group_by(
                ChatSession.session_id
            ).all()
            
            avg_conversation_length = len(session_lengths) / max(global_stats.get('total_sessions', 1), 1)
            
            # Tool usage from global stats
            tool_usage = global_stats.get('tool_usage', {})
            
            # Average response time by tool
            avg_response_time_by_tool = {}
            for tool, times in self.tool_response_times.items():
                if times:
                    avg_response_time_by_tool[tool] = sum(times) / len(times)
            
            # Success rate (assume high if no errors tracked)
            success_rate = max(0, 100 - (self.error_count / max(self.total_requests, 1)) * 100)
            
            db.close()
            
            metrics = ChatbotMetrics(
                timestamp=datetime.utcnow().isoformat(),
                total_conversations=global_stats.get('total_messages', 0),
                conversations_last_hour=conversations_last_hour,
                conversations_last_day=conversations_last_day,
                avg_conversation_length=avg_conversation_length,
                tool_usage=tool_usage,
                avg_response_time_by_tool=avg_response_time_by_tool,
                success_rate=success_rate,
                user_satisfaction=85.0  # Placeholder - could be calculated from feedback
            )
            
            self.metrics_history.append(('chatbot', asdict(metrics)))
            
        except Exception as e:
            logger.error(f"Error collecting chatbot metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            metrics_file = self.metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Save daily metrics file
            with open(metrics_file, 'w') as f:
                json.dump(list(self.metrics_history), f, indent=2, default=str)
            
            # Save latest metrics for dashboard
            latest_file = self.metrics_dir / "latest.json"
            latest_metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'system': None,
                'chatbot': None
            }
            
            # Get latest metrics of each type
            for metric_type, data in reversed(self.metrics_history):
                if metric_type == 'system' and latest_metrics['system'] is None:
                    latest_metrics['system'] = data
                elif metric_type == 'chatbot' and latest_metrics['chatbot'] is None:
                    latest_metrics['chatbot'] = data
                
                if latest_metrics['system'] and latest_metrics['chatbot']:
                    break
            
            with open(latest_file, 'w') as f:
                json.dump(latest_metrics, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def record_request(self, response_time: float, tool_used: str = None, error: bool = False):
        """Record a request with its response time and tool used"""
        self.total_requests += 1
        self.request_times.append(response_time)
        
        if tool_used:
            self.tool_response_times[tool_used].append(response_time)
            # Keep only recent 100 times per tool
            if len(self.tool_response_times[tool_used]) > 100:
                self.tool_response_times[tool_used] = self.tool_response_times[tool_used][-100:]
        
        if error:
            self.error_count += 1
    
    def add_active_session(self, session_id: str):
        """Add an active session"""
        self.active_sessions.add(session_id)
    
    def remove_active_session(self, session_id: str):
        """Remove an active session"""
        self.active_sessions.discard(session_id)
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the latest metrics"""
        if not self.metrics_history:
            return {}
        
        latest = {'system': None, 'chatbot': None}
        for metric_type, data in reversed(self.metrics_history):
            if metric_type not in latest or latest[metric_type] is None:
                latest[metric_type] = data
        
        return latest
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        relevant_metrics = [
            (metric_type, data) for metric_type, data in self.metrics_history
            if datetime.fromisoformat(data['timestamp']) > cutoff_time
        ]
        
        if not relevant_metrics:
            return {}
        
        # Aggregate system metrics
        system_metrics = [data for metric_type, data in relevant_metrics if metric_type == 'system']
        chatbot_metrics = [data for metric_type, data in relevant_metrics if metric_type == 'chatbot']
        
        summary = {
            'period_hours': hours,
            'data_points': len(relevant_metrics),
            'system': self._aggregate_system_metrics(system_metrics),
            'chatbot': self._aggregate_chatbot_metrics(chatbot_metrics)
        }
        
        return summary
    
    def _aggregate_system_metrics(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate system metrics"""
        if not metrics:
            return {}
        
        return {
            'avg_cpu_percent': sum(m['cpu_percent'] for m in metrics) / len(metrics),
            'max_cpu_percent': max(m['cpu_percent'] for m in metrics),
            'avg_memory_percent': sum(m['memory_percent'] for m in metrics) / len(metrics),
            'max_memory_percent': max(m['memory_percent'] for m in metrics),
            'avg_response_time': sum(m['avg_response_time'] for m in metrics) / len(metrics),
            'total_requests': metrics[-1]['total_requests'] if metrics else 0,
            'avg_error_rate': sum(m['error_rate'] for m in metrics) / len(metrics)
        }
    
    def _aggregate_chatbot_metrics(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate chatbot metrics"""
        if not metrics:
            return {}
        
        # Combine tool usage
        combined_tool_usage = defaultdict(int)
        for m in metrics:
            for tool, count in m.get('tool_usage', {}).items():
                combined_tool_usage[tool] += count
        
        return {
            'total_conversations': metrics[-1]['total_conversations'] if metrics else 0,
            'conversations_last_hour': metrics[-1]['conversations_last_hour'] if metrics else 0,
            'avg_conversation_length': sum(m['avg_conversation_length'] for m in metrics) / len(metrics),
            'tool_usage': dict(combined_tool_usage),
            'avg_success_rate': sum(m['success_rate'] for m in metrics) / len(metrics),
            'avg_user_satisfaction': sum(m['user_satisfaction'] for m in metrics) / len(metrics)
        }

class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'error_rate': 5.0,
            'avg_response_time': 10.0
        }
        self.alerts_history = deque(maxlen=100)
        self.last_alert_times = {}
        self.alert_cooldown = 300  # 5 minutes
    
    def check_alerts(self):
        """Check for alert conditions"""
        latest_metrics = self.metrics_collector.get_latest_metrics()
        system_metrics = latest_metrics.get('system', {})
        
        alerts = []
        
        for metric, threshold in self.alert_thresholds.items():
            if metric in system_metrics:
                value = system_metrics[metric]
                if value > threshold:
                    alert_key = f"{metric}_{threshold}"
                    
                    # Check cooldown
                    last_alert = self.last_alert_times.get(alert_key, 0)
                    if time.time() - last_alert > self.alert_cooldown:
                        alert = {
                            'timestamp': datetime.utcnow().isoformat(),
                            'type': 'threshold_exceeded',
                            'metric': metric,
                            'value': value,
                            'threshold': threshold,
                            'severity': self._get_severity(metric, value, threshold)
                        }
                        alerts.append(alert)
                        self.alerts_history.append(alert)
                        self.last_alert_times[alert_key] = time.time()
        
        return alerts
    
    def _get_severity(self, metric: str, value: float, threshold: float) -> str:
        """Determine alert severity"""
        excess = (value - threshold) / threshold * 100
        
        if excess > 50:
            return 'critical'
        elif excess > 25:
            return 'high'
        elif excess > 10:
            return 'medium'
        else:
            return 'low'
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alerts_history
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]

# Global metrics collector instance
metrics_collector = MetricsCollector()
alert_manager = AlertManager(metrics_collector)

def start_monitoring():
    """Start the monitoring system"""
    metrics_collector.start_collection()
    logger.info("Monitoring system started")

def stop_monitoring():
    """Stop the monitoring system"""
    metrics_collector.stop_collection()
    logger.info("Monitoring system stopped")

def get_dashboard_data() -> Dict[str, Any]:
    """Get data for monitoring dashboard"""
    return {
        'latest_metrics': metrics_collector.get_latest_metrics(),
        'summary_24h': metrics_collector.get_metrics_summary(24),
        'summary_1h': metrics_collector.get_metrics_summary(1),
        'recent_alerts': alert_manager.get_recent_alerts(24),
        'system_health': _get_system_health_status()
    }

def _get_system_health_status() -> str:
    """Get overall system health status"""
    latest_metrics = metrics_collector.get_latest_metrics()
    system_metrics = latest_metrics.get('system', {})
    
    if not system_metrics:
        return 'unknown'
    
    # Check critical thresholds
    if (system_metrics.get('cpu_percent', 0) > 90 or 
        system_metrics.get('memory_percent', 0) > 95 or
        system_metrics.get('error_rate', 0) > 10):
        return 'critical'
    
    # Check warning thresholds
    if (system_metrics.get('cpu_percent', 0) > 70 or 
        system_metrics.get('memory_percent', 0) > 80 or
        system_metrics.get('error_rate', 0) > 2):
        return 'warning'
    
    return 'healthy'

if __name__ == "__main__":
    # Test monitoring system
    print("Starting monitoring test...")
    
    start_monitoring()
    
    # Simulate some requests
    for i in range(10):
        metrics_collector.record_request(
            response_time=1.5 + i * 0.1,
            tool_used=['rag', 'web_search', 'math'][i % 3],
            error=(i % 5 == 0)
        )
    
    time.sleep(5)
    
    # Get dashboard data
    dashboard_data = get_dashboard_data()
    print("Dashboard data collected:")
    print(json.dumps(dashboard_data, indent=2, default=str))
    
    stop_monitoring()
    print("Monitoring test completed")