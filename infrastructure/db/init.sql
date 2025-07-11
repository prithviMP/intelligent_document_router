
-- Database initialization script for Intelligent Document Router

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    original_name VARCHAR(255) NOT NULL,
    storage_path TEXT NOT NULL,
    file_size INTEGER,
    mime_type VARCHAR(100),
    doc_type VARCHAR(50),
    confidence REAL DEFAULT 0.0,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    tenant_id UUID DEFAULT uuid_generate_v4()
);

-- Metadata table
CREATE TABLE IF NOT EXISTS metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    key_entities JSONB DEFAULT '{}',
    extracted_text TEXT,
    summary TEXT,
    topics TEXT[],
    sentiment_score REAL DEFAULT 0.0,
    language VARCHAR(10) DEFAULT 'en',
    risk_score REAL DEFAULT 0.0,
    compliance_flags TEXT[],
    related_documents UUID[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Routing decisions table
CREATE TABLE IF NOT EXISTS routing_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    assigned_to VARCHAR(255) NOT NULL,
    assigned_team VARCHAR(100) NOT NULL,
    priority INTEGER DEFAULT 3,
    confidence REAL DEFAULT 0.0,
    rule_id VARCHAR(100),
    routing_reason TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    escalation_deadline TIMESTAMP WITH TIME ZONE,
    escalated_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Processing queue table
CREATE TABLE IF NOT EXISTS processing_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    service_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    priority INTEGER DEFAULT 3,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    payload JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    scheduled_at TIMESTAMP WITH TIME ZONE,
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    team VARCHAR(100),
    role VARCHAR(50) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- Routing rules table
CREATE TABLE IF NOT EXISTS routing_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    rule_id VARCHAR(100) UNIQUE NOT NULL,
    conditions JSONB NOT NULL,
    actions JSONB NOT NULL,
    priority INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tenant_id UUID DEFAULT uuid_generate_v4()
);

-- Workflow executions table
CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    workflow_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    execution_log JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);

CREATE INDEX IF NOT EXISTS idx_routing_assigned_to ON routing_decisions(assigned_to);
CREATE INDEX IF NOT EXISTS idx_routing_team ON routing_decisions(assigned_team);
CREATE INDEX IF NOT EXISTS idx_routing_priority ON routing_decisions(priority);
CREATE INDEX IF NOT EXISTS idx_routing_status ON routing_decisions(status);
CREATE INDEX IF NOT EXISTS idx_routing_created_at ON routing_decisions(created_at);

CREATE INDEX IF NOT EXISTS idx_queue_service_status ON processing_queue(service_name, status);
CREATE INDEX IF NOT EXISTS idx_queue_priority ON processing_queue(priority);
CREATE INDEX IF NOT EXISTS idx_queue_scheduled ON processing_queue(scheduled_at);

CREATE INDEX IF NOT EXISTS idx_user_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_user_team ON users(team);
CREATE INDEX IF NOT EXISTS idx_user_role ON users(role);

CREATE INDEX IF NOT EXISTS idx_rule_active ON routing_rules(is_active);
CREATE INDEX IF NOT EXISTS idx_rule_priority ON routing_rules(priority);

CREATE INDEX IF NOT EXISTS idx_workflow_status ON workflow_executions(status);
CREATE INDEX IF NOT EXISTS idx_workflow_name ON workflow_executions(workflow_name);

-- Insert sample data
INSERT INTO users (email, username, full_name, team, role) VALUES
('john.doe@company.com', 'john.doe', 'John Doe', 'legal_team', 'user'),
('jane.smith@company.com', 'jane.smith', 'Jane Smith', 'legal_team', 'user'),
('alice.johnson@company.com', 'alice.johnson', 'Alice Johnson', 'finance_team', 'user'),
('bob.wilson@company.com', 'bob.wilson', 'Bob Wilson', 'finance_team', 'user'),
('ceo@company.com', 'ceo', 'CEO', 'management_team', 'admin'),
('cfo@company.com', 'cfo', 'CFO', 'management_team', 'admin'),
('hr.manager@company.com', 'hr.manager', 'HR Manager', 'hr_team', 'user'),
('hr.assistant@company.com', 'hr.assistant', 'HR Assistant', 'hr_team', 'user'),
('support@company.com', 'support', 'Support Team', 'general_team', 'user')
ON CONFLICT (email) DO NOTHING;

-- Insert default routing rules
INSERT INTO routing_rules (name, rule_id, conditions, actions, priority) VALUES
('Contract Documents to Legal Team', 'contract_legal', 
 '{"doc_type": ["contract", "agreement", "legal_document"], "confidence": {"min": 0.8}}',
 '{"assign_to_team": "legal_team", "priority": 1, "escalation_hours": 24}', 1),
('Invoices to Finance Team', 'invoice_finance',
 '{"doc_type": ["invoice", "bill", "payment_request"], "confidence": {"min": 0.7}}',
 '{"assign_to_team": "finance_team", "priority": 2, "escalation_hours": 48}', 2),
('Reports to Management', 'report_management',
 '{"doc_type": ["report", "analysis", "summary"], "confidence": {"min": 0.6}}',
 '{"assign_to_team": "management_team", "priority": 3, "escalation_hours": 72}', 3),
('HR Documents to HR Team', 'hr_documents',
 '{"doc_type": ["resume", "application", "hr_form"], "confidence": {"min": 0.7}}',
 '{"assign_to_team": "hr_team", "priority": 2, "escalation_hours": 48}', 4),
('General Documents', 'fallback_general',
 '{"doc_type": ["*"], "confidence": {"min": 0.0}}',
 '{"assign_to_team": "general_team", "priority": 4, "escalation_hours": 96}', 9999)
ON CONFLICT (rule_id) DO NOTHING;

-- Create triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_metadata_updated_at BEFORE UPDATE ON metadata FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_routing_decisions_updated_at BEFORE UPDATE ON routing_decisions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_processing_queue_updated_at BEFORE UPDATE ON processing_queue FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_routing_rules_updated_at BEFORE UPDATE ON routing_rules FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_workflow_executions_updated_at BEFORE UPDATE ON workflow_executions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
