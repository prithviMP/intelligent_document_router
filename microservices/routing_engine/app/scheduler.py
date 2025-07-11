
"""
Background task scheduler for routing engine
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import and_

from .main import SessionLocal, RoutingDecision, routing_engine, send_notification

logger = logging.getLogger(__name__)

class RoutingScheduler:
    """Background scheduler for routing tasks"""
    
    def __init__(self):
        self.running = False
    
    async def start(self):
        """Start the scheduler"""
        self.running = True
        logger.info("Routing scheduler started")
        
        while self.running:
            try:
                await self.check_escalations()
                await self.cleanup_old_decisions()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Scheduler error: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def check_escalations(self):
        """Check for documents that need escalation"""
        try:
            db = SessionLocal()
            try:
                # Find documents past escalation deadline
                now = datetime.utcnow()
                overdue_decisions = db.query(RoutingDecision).filter(
                    and_(
                        RoutingDecision.escalation_deadline < now,
                        RoutingDecision.status.in_(["pending", "in_progress"])
                    )
                ).all()
                
                for decision in overdue_decisions:
                    await self.escalate_decision(decision, db)
                
                if overdue_decisions:
                    db.commit()
                    logger.info(f"Escalated {len(overdue_decisions)} overdue documents")
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error checking escalations: {str(e)}")
    
    async def escalate_decision(self, decision: RoutingDecision, db: Session):
        """Escalate a routing decision"""
        try:
            # Move to management team with highest priority
            management_workload = routing_engine._get_team_workload("management_team", db)
            new_assignee = routing_engine._assign_to_user("management_team", management_workload)
            
            decision.assigned_to = new_assignee
            decision.assigned_team = "management_team"
            decision.priority = 1
            decision.status = "escalated"
            decision.routing_reason = f"Auto-escalated: Deadline exceeded"
            decision.updated_at = datetime.utcnow()
            
            # Send notification
            await send_notification(decision)
            
            logger.info(f"Auto-escalated document {decision.document_id} to {new_assignee}")
            
        except Exception as e:
            logger.error(f"Error escalating decision {decision.document_id}: {str(e)}")
    
    async def cleanup_old_decisions(self):
        """Clean up old completed decisions"""
        try:
            db = SessionLocal()
            try:
                # Delete decisions older than 90 days that are completed
                ninety_days_ago = datetime.utcnow() - timedelta(days=90)
                deleted_count = db.query(RoutingDecision).filter(
                    and_(
                        RoutingDecision.created_at < ninety_days_ago,
                        RoutingDecision.status == "completed"
                    )
                ).delete()
                
                if deleted_count > 0:
                    db.commit()
                    logger.info(f"Cleaned up {deleted_count} old routing decisions")
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error cleaning up old decisions: {str(e)}")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("Routing scheduler stopped")

# Global scheduler instance
scheduler = RoutingScheduler()
