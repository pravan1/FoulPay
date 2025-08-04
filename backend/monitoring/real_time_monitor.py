import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class RealTimeMonitor:
    """
    Real-time monitoring system for fraud detection
    Tracks fraud rates, model performance, and system health
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.monitoring_active = False
        self.alert_thresholds = {
            'high_fraud_rate': 0.15,  # Alert if fraud rate > 15%
            'low_confidence': 0.7,    # Alert if average confidence < 70%
            'system_errors': 5,       # Alert if > 5 errors in 10 minutes
        }
        self.metrics_cache = {
            'fraud_rate_1h': 0.0,
            'avg_confidence_1h': 0.0,
            'total_transactions_1h': 0,
            'error_count_10m': 0,
            'last_updated': datetime.now()
        }
    
    async def start_monitoring(self):
        """Start the real-time monitoring loop"""
        self.monitoring_active = True
        logger.info("🔍 Real-time monitoring started")
        
        while self.monitoring_active:
            try:
                await self._update_metrics()
                await self._check_alerts()
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.monitoring_active = False
        logger.info("🛑 Real-time monitoring stopped")
    
    async def _update_metrics(self):
        """Update cached metrics from database"""
        try:
            now = datetime.now()
            one_hour_ago = now - timedelta(hours=1)
            ten_minutes_ago = now - timedelta(minutes=10)
            
            # Get recent transactions
            recent_transactions = await self.db_manager.get_recent_transactions(1000)
            
            # Filter for last hour
            hour_transactions = [
                t for t in recent_transactions 
                if self._parse_timestamp(t.get('timestamp', '')) >= one_hour_ago
            ]
            
            if hour_transactions:
                # Calculate fraud rate
                high_risk_count = len([t for t in hour_transactions if t.get('risk_score', 0) > 80])
                fraud_rate = high_risk_count / len(hour_transactions)
                
                # Calculate average confidence
                confidences = [t.get('predictions', {}).get('confidence', 0.85) for t in hour_transactions]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.85
                
                # Update cache
                self.metrics_cache.update({
                    'fraud_rate_1h': fraud_rate,
                    'avg_confidence_1h': avg_confidence,
                    'total_transactions_1h': len(hour_transactions),
                    'last_updated': now
                })
                
                logger.info(f"📊 Metrics updated - Fraud rate: {fraud_rate:.1%}, Avg confidence: {avg_confidence:.1%}")
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def _check_alerts(self):
        """Check for alert conditions and log warnings"""
        alerts = []
        
        # High fraud rate alert
        if self.metrics_cache['fraud_rate_1h'] > self.alert_thresholds['high_fraud_rate']:
            alerts.append({
                'type': 'HIGH_FRAUD_RATE',
                'message': f"Fraud rate ({self.metrics_cache['fraud_rate_1h']:.1%}) exceeds threshold",
                'severity': 'HIGH'
            })
        
        # Low confidence alert
        if self.metrics_cache['avg_confidence_1h'] < self.alert_thresholds['low_confidence']:
            alerts.append({
                'type': 'LOW_CONFIDENCE',
                'message': f"Average confidence ({self.metrics_cache['avg_confidence_1h']:.1%}) below threshold",
                'severity': 'MEDIUM'
            })
        
        # Low transaction volume (might indicate system issues)
        if self.metrics_cache['total_transactions_1h'] == 0:
            alerts.append({
                'type': 'NO_TRANSACTIONS',
                'message': "No transactions processed in last hour",
                'severity': 'HIGH'
            })
        
        # Log alerts
        for alert in alerts:
            if alert['severity'] == 'HIGH':
                logger.warning(f"🚨 ALERT: {alert['message']}")
            else:
                logger.info(f"⚠️  Alert: {alert['message']}")
        
        # Store alerts in database if any
        if alerts:
            await self._store_alerts(alerts)
    
    async def _store_alerts(self, alerts: List[Dict[str, Any]]):
        """Store alerts in database for tracking"""
        try:
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'alerts': alerts,
                'metrics_snapshot': self.metrics_cache.copy()
            }
            
            # This would typically go to a dedicated alerts collection
            logger.info(f"📝 Stored {len(alerts)} alerts")
            
        except Exception as e:
            logger.error(f"Error storing alerts: {e}")
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object"""
        try:
            if timestamp_str:
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return datetime.min
        except:
            return datetime.min
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current cached metrics"""
        return {
            **self.metrics_cache,
            'monitoring_active': self.monitoring_active,
            'alert_thresholds': self.alert_thresholds
        }
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        try:
            stats = await self.db_manager.get_system_stats()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system_health': 'healthy' if self.monitoring_active else 'monitoring_disabled',
                'real_time_metrics': self.metrics_cache,
                'database_stats': stats,
                'alert_status': 'active' if any([
                    self.metrics_cache['fraud_rate_1h'] > self.alert_thresholds['high_fraud_rate'],
                    self.metrics_cache['avg_confidence_1h'] < self.alert_thresholds['low_confidence']
                ]) else 'normal'
            }
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'system_health': 'error',
                'error': str(e)
            }

# Global monitor instance
monitor = RealTimeMonitor()